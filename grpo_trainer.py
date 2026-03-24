import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import copy
import random
import numpy as np
import sys
import os
import jiwer
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from avdataset import GRIDDataset, CMLRDataset, BucketBatchSampler
from avmodel import AVSRModel
from constants import *


class JointLoss(nn.Module):
    def __init__(self, mrt_alpha=0.05, grpo_eps=0.2, grpo_beta=0.01):
        super().__init__()
        self.mrt_alpha = mrt_alpha
        # GRPO 超参数
        self.grpo_eps = grpo_eps  # PPO 裁剪参数
        self.grpo_beta = grpo_beta        # KL 惩罚系数

    def compute_mrt_loss(self, log_probs, risks):
        """
        log_probs: [Batch, N_samples] (包含 N-best 和 GT)
        risks: [Batch, N_samples] (WER, GT 的 risk 为 0)
        """
        # 1. 归一化概率分布 (Risk Minimization 需要相对概率)
        # alpha 用于平滑分布
        probs = F.softmax(log_probs * self.mrt_alpha, dim=-1)
        # 2. 计算预期风险: sum(P(y) * Risk(y))
        loss = torch.sum(probs * risks, dim=-1)
        return loss.mean()

    def compute_grpo_loss(self, policy_logps, old_policy_logps, rewards, advantages=None, use_clip=True):
        """
        GRPO (Group Relative Policy Optimization) 核心实现
        
        Args:
            policy_logps: [Batch, Group_Size] 当前策略的 log probs
            old_policy_logps: [Batch, Group_Size] 旧策略的 log probs (用于重要性采样)
            rewards: [Batch, Group_Size] 奖励值 (例如 -WER)
            advantages: [Batch, Group_Size] 优势函数 (可选，如果不提供则自动计算)
            use_clip: 是否使用 PPO 裁剪
            
        Returns:
            loss: GRPO 损失
            metrics: 包含 KL 散度等信息的字典
        """
        batch_size, group_size = policy_logps.shape
        
        # 计算重要性采样比率
        ratio = torch.exp(policy_logps - old_policy_logps)  # [Batch, Group_Size]
        
        # 如果没有提供优势，通过组内标准化计算
        if advantages is None:
            # GRPO: 使用组内相对奖励作为优势
            # 对每个样本的组内奖励进行标准化
            mean_rewards = rewards.mean(dim=-1, keepdim=True)  # [Batch, 1]
            std_rewards = rewards.std(dim=-1, unbiased=False, keepdim=True) + 1e-8  # [Batch, 1]
            advantages = (rewards - mean_rewards) / std_rewards  # [Batch, Group_Size]
        
        # 计算策略比率 * 优势
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.grpo_eps, 1 + self.grpo_eps) * advantages
        
        # PPO 裁剪目标
        if use_clip:
            policy_loss = -torch.min(surr1, surr2).mean()
        else:
            policy_loss = -(ratio * advantages).mean()
        
        # KL 惩罚项 (可选，用于防止策略偏离太远)
        # 使用近似 KL: log(pi_old / pi) = old_logp - logp
        approx_kl = (old_policy_logps - policy_logps).mean()
        kl_penalty = self.grpo_beta * approx_kl.abs()
        
        total_loss = policy_loss + kl_penalty
        
        metrics = {
            'policy_loss': policy_loss.item(),
            'kl_penalty': kl_penalty.item(),
            'approx_kl': approx_kl.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_ratio': ratio.mean().item()
        }
        
        return total_loss, metrics



class BaseTrainer:
    def __init__(self, model, optimizer, lr_scheduler=None, accumulate_step=1, device='cuda:0'):
        self.model = model
        self.model.train()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.current_step = 0
        self.accumulate_step = accumulate_step
    
    def train_step(self, vid_inp, aud_inp, targets, vid_lens, aud_lens, tgt_lens):
        ''' 
        optimizer.zero_grad()
        #losses = model(vid_inp, aud_inp, targets, vid_lens, aud_lens, tgt_lens)
        losses = model(clean_vid_inp, noisy_vid_inp, clean_aud_inp, noisy_aud_inp, targets, vid_lens, aud_lens, tgt_lens)
        loss = losses['avsr'] + losses['drl']
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        '''
        loss = self.model(vid_inp, aud_inp, targets, vid_lens, aud_lens, tgt_lens)[0]
        loss = loss / self.accumulate_step
        loss.backward()
        self.current_step += 1
        if self.current_step % self.accumulate_step == 0:
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.current_step = 0
        return {"avsr_loss": loss.data.item()}



class MRT_GRPO_Trainer:
    def __init__(self, model, optimizer, tokenizer, lambda_ce=0.2, lambda_mrt=1.0, lambda_grpo=0.1, device='cuda:0', grpo_group_size=8):
        self.model = model
        self.model.train()
        self.optimizer = optimizer
        self.tokenizer = tokenizer   # 用于计算 WER 时的解码
        self.pad_id, self.bos_id, self.eos_id = (tokenizer.index(x) for x in [PAD, BOS, EOS])
        print(f'PAD ID: {self.pad_id}, BOS ID: {self.bos_id}, EOS ID: {self.eos_id}')
        self.device = device
        
        # GRPO 不需要参考模型！这是与 DPO 的主要区别
        # 但我们可能需要存储旧策略用于重要性采样
        self.old_model = None
        
        self.criterion = JointLoss(mrt_alpha=1., grpo_eps=0.2, grpo_beta=0.01)
        self.weights = {'ce': lambda_ce, 'mrt': lambda_mrt, 'grpo': lambda_grpo}
        
        # GRPO 组大小 (每个样本采样的候选数量)
        self.grpo_group_size = grpo_group_size
        
        self.current_step = 0

    def get_batch_log_probs(self, model, tgt_tokens, enc_memory, src_lens, tgt_lens, len_norm=True):
        """
            通用辅助函数：计算给定 Token 序列在模型下的 Log-Prob (Sum over sequence)
        """
        # Transformer Decoder Forward
        # tgt_tokens: [Batch, Len] (包含 <sos>, <eos>)
        # enc_memory: Encoder 输出 [Batch, T, Dim]
        
        # 假设 model.decoder_forward 返回 logits [Batch, Len, Vocab]
        # 注意：这里调用的是 Teacher Forcing 模式
        all_log_probs = model.decoder_forward(tgt_tokens, enc_memory, src_lens, tgt_lens)
        
        # Gather 对应 Token 的概率
        # input: <sos> A B C ...
        # target: A B C <eos> ...
        # 通常 Transformer Decoder 输出对齐是 shift right 的
        # 这里假设 logits 已经对应了我们要预测的位置
        
        # 简单处理：取 tokens[:, 1:] 作为目标，logits[:, :-1, :] 作为预测
        targets = tgt_tokens[:, 1:]
        preds_log_probs = all_log_probs[:, :-1, :]
        
        target_log_probs = torch.gather(preds_log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        # Mask padding (假设 pad_id = 0)
        pad_mask = (targets != 0).float()
        
        # Sum log probs over sequence length
        seq_log_probs = (target_log_probs * pad_mask).sum(dim=-1)
       
        if len_norm:
            seq_lens = pad_mask.sum(dim=-1).clamp(min=1)
            return seq_log_probs / seq_lens

        return seq_log_probs

    def compute_rewards(self, hyp_texts, ref_text):
        """
        计算奖励：使用负 WER (越高越好)
        也可以加入其他奖励，如长度惩罚等
        """
        rewards = []
        for hyp in hyp_texts:
            wer = jiwer.wer(ref_text, hyp)
            # 奖励 = -WER，范围通常在 [-1, 0] 之间
            # 可以添加缩放因子
            reward = -wer
            rewards.append(reward)
        return rewards

    def update_old_policy(self):
        """
        更新旧策略模型（用于 GRPO 的重要性采样）
        可以选择在每个 epoch 开始时更新，或每隔几步更新
        """
        # 深拷贝当前模型作为旧策略
        self.old_model = copy.deepcopy(self.model)
        for param in self.old_model.parameters():
            param.requires_grad = False
        self.old_model.eval()

    def train_step(self, vid_inp, aud_inp, tgt_tokens, vid_lens, aud_lens, tgt_lens, n_best=5):
        """
        vid_inp: [B, C, T, H, W]
        aud_inp: [B, T_a, D]
        tgt_tokens: [B, L]
        """
        batch_size = vid_inp.size(0)
        
        # -------------------------------------------
        # Step 0: 更新旧策略 (如果需要)
        # -------------------------------------------
        # 策略：每隔一定步数更新旧策略，或每个 batch 都更新
        # 这里选择每个 batch 都更新（更稳定但慢），可以优化为每隔 N 步
        if self.old_model is None:
            self.update_old_policy()
        
        # -------------------------------------------
        # Step 1: Encoder 前向 (只做一次)
        # -------------------------------------------
        # 联合编码 AV 特征
        enc_memory, src_lens = self.model.encode_av(vid_inp, aud_inp, vid_lens, aud_lens)

        # -------------------------------------------
        # Step 2: 采样与构建数据 (No Grad)
        # -------------------------------------------
        with torch.no_grad():
            # Beam Search 产生 N-best
            # 返回: List[List[Dict{'tokens': Tensor, 'text': str}]]
            nbest_batch, tgt_text = [], []
            nbest_output = self.model.generate(enc_memory, src_lens, self.bos_id, self.eos_id, max_dec_len=50, beam_size=n_best)
            for preds, tgts in zip(nbest_output, tgt_tokens):
                batch_out = []
                for pred in preds:
                    txt = ''.join([self.tokenizer[i] for i in pred.tolist() if i not in [self.pad_id, self.bos_id, self.eos_id]])
                    batch_out.append({'text': txt, 'tokens': torch.cat([torch.tensor([self.bos_id], device=pred.device), pred])})
                nbest_batch.append(batch_out)
                
                tgt_text.append(''.join([self.tokenizer[i] for i in tgts.tolist() if i not in [self.pad_id, self.bos_id, self.eos_id]]))
            
            # 构建 GRPO 数据
            grpo_tokens_list = []   # 用于 GRPO
            grpo_rewards_list = []  # 用于 GRPO
            mrt_tokens_list = []    # 用于 MRT
            mrt_risks_list = []     # 用于 MRT
            
            for b in range(batch_size):
                hyps = nbest_batch[b]  # N 个候选
                ref_text = tgt_text[b]
                
                # 计算 WER 和奖励
                wers = []
                rewards = []
                current_batch_tokens = []
                
                for h in hyps:
                    wer = jiwer.wer(ref_text, h['text'])
                    wers.append(wer)
                    rewards.append(-wer)  # 奖励 = -WER   # 8.70 -> 8.49 (n_best = groups = 5)
                    #rewards.append(np.exp(-wer))  # 奖励 = exp(-WER)
                    current_batch_tokens.append(h['tokens'])
                
                # --- 构建 GRPO 数据 ---
                # 选择 group_size 个样本（可以包括 GT）
                # 策略：选择 WER 分布不同的样本以获得更多学习信号
                group_size = min(self.grpo_group_size, len(hyps))
                
                # 可以按 WER 分层采样，或直接取前 N 个
                # 这里简单取前 group_size 个
                selected_indices = list(range(group_size))
                grpo_tokens_list.extend([current_batch_tokens[i] for i in selected_indices])
                grpo_rewards_list.extend([rewards[i] for i in selected_indices])
                
                # --- 构建 MRT 数据 ---
                # 技巧：将 GT (Risk=0) 注入到 N-best 列表末尾
                current_batch_tokens.append(tgt_tokens[b])   # 加入 GT
                wers.append(0.0)   # GT 的 Risk 为 0
                
                mrt_tokens_list.extend(current_batch_tokens)
                mrt_risks_list.extend(wers)

            # Pad 序列以进行批处理
            # MRT inputs: [Batch * (N+1), Max_Len]
            mrt_inputs = pad_sequence(mrt_tokens_list, batch_first=True, padding_value=0).to(self.device)
            mrt_risks = torch.tensor(mrt_risks_list).view(batch_size, n_best + 1).to(self.device)
            
            # GRPO inputs: [Batch * Group_Size, Max_Len]
            grpo_inputs = pad_sequence(grpo_tokens_list, batch_first=True, padding_value=0).to(self.device)
            grpo_rewards = torch.tensor(grpo_rewards_list).view(batch_size, group_size).to(self.device)

        # -------------------------------------------
        # Step 3: 当前策略计算 Log-Probs (With Grad)
        # -------------------------------------------
        # A. 计算 MRT 需要的所有序列概率 (N-best + GT)
        mrt_mem_expanded = enc_memory.repeat_interleave(n_best + 1, dim=0)
        src_lens_expanded = src_lens.repeat_interleave(n_best + 1, dim=0)
        tgt_lens_expanded = tgt_lens.repeat_interleave(n_best + 1, dim=0)
        
        all_mrt_logps = self.get_batch_log_probs(self.model, mrt_inputs, mrt_mem_expanded, src_lens_expanded, tgt_lens_expanded)
        all_mrt_logps_view = all_mrt_logps.view(batch_size, n_best + 1)
        
        # B. 计算 GRPO 需要的组内概率
        grpo_mem_expanded = enc_memory.repeat_interleave(group_size, dim=0)
        grpo_src_lens_expanded = src_lens.repeat_interleave(group_size, dim=0)
        grpo_tgt_lens_expanded = tgt_lens.repeat_interleave(group_size, dim=0)
        
        policy_grpo_logps = self.get_batch_log_probs(self.model, grpo_inputs, grpo_mem_expanded, grpo_src_lens_expanded, grpo_tgt_lens_expanded)
        policy_grpo_logps = policy_grpo_logps.view(batch_size, group_size)

        # -------------------------------------------
        # Step 4: 旧策略计算 Log-Probs (No Grad)
        # -------------------------------------------
        with torch.no_grad():
            # 使用旧模型计算概率用于重要性采样
            old_enc_memory, old_src_lens = self.old_model.encode_av(vid_inp, aud_inp, vid_lens, aud_lens)
            old_grpo_mem_expanded = old_enc_memory.repeat_interleave(group_size, dim=0)
            old_grpo_src_lens_expanded = old_src_lens.repeat_interleave(group_size, dim=0)
            
            old_grpo_logps = self.get_batch_log_probs(self.old_model, grpo_inputs, old_grpo_mem_expanded, old_grpo_src_lens_expanded, grpo_tgt_lens_expanded)
            old_grpo_logps = old_grpo_logps.view(batch_size, group_size)

        # -------------------------------------------
        # Step 5: 计算联合损失
        # -------------------------------------------
        # 1. MRT Loss
        loss_mrt = self.criterion.compute_mrt_loss(all_mrt_logps_view, mrt_risks)
        
        # 2. GRPO Loss
        loss_grpo, grpo_metrics = self.criterion.compute_grpo_loss(
            policy_grpo_logps, old_grpo_logps, grpo_rewards
        )
        
        # 3. CE Loss (Regularization)
        # GT 的 log prob 在 MRT 列表的最后一个
        policy_chosen_logps = all_mrt_logps_view[:, -1]
        loss_ce = -policy_chosen_logps.mean()

        total_loss = (self.weights['mrt'] * loss_mrt + 
                      self.weights['grpo'] * loss_grpo + 
                      self.weights['ce'] * loss_ce)

        self.model.zero_grad()

        total_loss.backward()
        
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()

        # 可选：每隔几步更新旧策略
        self.current_step += 1
        if self.current_step % 10 == 0:
            self.update_old_policy()
            self.current_step = 0

        return {
            "total": total_loss.item(),
            "mrt": loss_mrt.item(),
            "grpo": loss_grpo.item(),
            "ce": loss_ce.item(),
            "grpo_policy_loss": grpo_metrics['policy_loss'],
            "grpo_kl": grpo_metrics['approx_kl'],
            "grpo_mean_adv": grpo_metrics['mean_advantage']
        }
        


def main():
    device = torch.device('cuda:' + str(sys.argv[1]))
    print('running device:', torch.cuda.get_device_name(), device)
    data_type = str(sys.argv[2]).strip().lower()  # grid or cmlr or lrs2/3
    if data_type == 'grid':
        data_root = r'../LipData/GRID/LIP_160_80/lip'
        train_set = GRIDDataset(data_root, r'data/unseen_train.json', phase='train', setting='unseen')
        #val_set = GRIDDataset(data_root, r'data/unseen_val.json', phase='test', setting='unseen')
    elif data_type == 'cmlr':
        data_root = r'../LipData/CMLR'
        train_set = CMLRDataset(data_root, r'data/unseen_train.csv', phase='train', setting='unseen')
        #val_set = CMLRDataset(data_root, r'data/unseen_test.csv', phase='test', setting='unseen')
    else:
        raise NotImplementedError('Unknown Dataset!!')
    
    #model_path = None
    model_path = 'grid_avg_10.pt'
    #model_path = 'checkpoints/grid/stage1_iter_49.pt'
    
    model = AVSRModel(len(train_set.vocab)).to(device)
    print('参数量(M)：', sum(param.numel() for param in model.parameters())/1e6)
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        states = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(states)
        print(f'loading weights from {model_path} ...')
    print(model)

    batch_size = 32
    accumulate_step = 1
    epochs = {'I': 50, 'II': 5}   
    #lrs = {'I': 3e-4, 'II': 5e-6}  # 1.61
    lrs = {'I': 3e-4, 'II': 1e-5}  # 1.61
    savedir = os.path.join('checkpoints', 'grid')

    '''
    train_set.noise_ratio = 0.25
    data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=train_set.collate_pad)  
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.AdamW([*model.avsr.parameters(), *model.spk.parameters()], lr=3 * lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lrs['I'], betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
    num_iters = len(data_loader) * epochs['I'] // accumulate_step
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_iters // 10, num_training_steps=num_iters)
    ## I AVSR预训练
    base_trainer = BaseTrainer(model, optimizer, lr_scheduler, accumulate_step=accumulate_step, device=device)
    for ep in range(epochs['I']):
        for i, batch_data in enumerate(data_loader):  # (B, T, C, H, W)
            #clean_vid_inp = batch_data['clean_vid'].to(device)
            #clean_aud_inp = batch_data['clean_aud'].to(device)
            noisy_vid_inp = batch_data['noisy_vid'].to(device)
            noisy_aud_inp = batch_data['noisy_aud'].to(device)
            targets = batch_data['txt'].to(device)
            vid_lens = batch_data['vid_lens'].to(device)
            aud_lens = batch_data['aud_lens'].to(device)
            tgt_lens = batch_data['txt_lens'].to(device)
            #print(batch_data['clean_vid_lens'], batch_data['noisy_vid_lens'], batch_data['clean_aud_lens'], batch_data['noisy_aud_lens'])
            loss_val = base_trainer.train_step(noisy_vid_inp, noisy_aud_inp, targets, vid_lens, aud_lens, tgt_lens)
            print(f'Epoch {ep}, Iter {i}, lr: {optimizer.param_groups[0]["lr"]}, loss: {loss_val["avsr_loss"]}', flush=True)
        if ep >= epochs['I'] - 10:
            savename = 'stage1_iter_{}.pt'.format(ep)
            if not os.path.exists(savedir): os.makedirs(savedir)
            save_path = os.path.join(savedir, savename)
            torch.save({'model': model.state_dict()}, save_path)
            print(f'Saved to {save_path}!!!', flush=True)
    '''
    

    train_set.noise_ratio = 0.
    data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False, collate_fn=train_set.collate_pad)  
    optimizer2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lrs['II'], betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
    ## II MRT-GRPO (替换原来的 MRT-DPO)
    mrt_grpo_trainer = MRT_GRPO_Trainer(model, optimizer2, train_set.vocab, 
                                        lambda_ce=0.2, lambda_mrt=0.8, lambda_grpo=0.2, 
                                        device=device, grpo_group_size=5)
    for ep in range(epochs['II']):
        for i, batch_data in enumerate(data_loader):  # (B, T, C, H, W)
            #clean_vid_inp = batch_data['clean_vid'].to(device)
            #clean_aud_inp = batch_data['clean_aud'].to(device)
            noisy_vid_inp = batch_data['noisy_vid'].to(device)
            noisy_aud_inp = batch_data['noisy_aud'].to(device)
            targets = batch_data['txt'].to(device)
            vid_lens = batch_data['vid_lens'].to(device)
            aud_lens = batch_data['aud_lens'].to(device)
            tgt_lens = batch_data['txt_lens'].to(device)
            #print(batch_data['clean_vid_lens'], batch_data['noisy_vid_lens'], batch_data['clean_aud_lens'], batch_data['noisy_aud_lens'])
            loss_val = mrt_grpo_trainer.train_step(noisy_vid_inp, noisy_aud_inp, targets, vid_lens, aud_lens, tgt_lens)
            print(f'Epoch {ep}, Iter {i}, lr: {optimizer2.param_groups[0]["lr"]}, '
                  f'mrt_loss: {loss_val["mrt"]}, grpo_loss: {loss_val["grpo"]}, '
                  f'ce_loss: {loss_val["ce"]}, total loss: {loss_val["total"]}', flush=True)
        if ep > 1:
            savename = 'stage2_iter_{}.pt'.format(ep)
            if not os.path.exists(savedir): os.makedirs(savedir)
            save_path = os.path.join(savedir, savename)
            torch.save({'model': model.state_dict()}, save_path)
            print(f'Saved to {save_path}!!!', flush=True)




@torch.no_grad()
def evaluate():
    device = torch.device('cuda:' + str(sys.argv[1]))
    print('running device:', torch.cuda.get_device_name(), device)
    data_type = str(sys.argv[2]).strip().lower()  # grid or cmlr or lrs2/3
    if data_type == 'grid':
        data_root = r'../LipData/GRID/LIP_160_80/lip'
        test_set = GRIDDataset(data_root, r'data/unseen_val.json', phase='test', setting='unseen')
    elif data_type == 'cmlr':
        data_root = r'../LipData/CMLR'
        test_set = CMLRDataset(data_root, r'data/unseen_test.csv', phase='test', setting='unseen')
    else:
        raise NotImplementedError('Unknown Dataset!!')
    
    batch_size = 32
    #model_path = 'grid_avg_10.pt'
    #model_path = 'checkpoints/grid/stage1_iter_49.pt'
    model_path = 'checkpoints/grid/stage2_iter_4.pt'
    
    model = AVSRModel(len(test_set.vocab)).to(device)
    print('参数量(M)：', sum(param.numel() for param in model.parameters())/1e6)
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        states = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.load_state_dict(states)
        print(f'loading weights from {model_path} ...')
    model.eval()
    print(len(test_set))

    data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=test_set.collate_pad)  
    preds, refs = [], []
    PAD_ID, BOS_ID, EOS_ID = (test_set.vocab.index(x) for x in [PAD, BOS, EOS])
    for batch_data in data_loader:
        #vid_inp = batch_data['vid'].to(device)
        #aud_inp = batch_data['aud'].to(device)
        vid_inp = batch_data['clean_vid'].to(device)
        #aud_inp = batch_data['clean_aud'].to(device)
        aud_inp = batch_data['noisy_aud'].to(device)
        tgt_txt = batch_data['txt'].to(device)
        #vid_lens = batch_data['clean_vid_lens'].to(device)
        #aud_lens = batch_data['clean_aud_lens'].to(device)
        vid_lens = batch_data['vid_lens'].to(device)
        aud_lens = batch_data['aud_lens'].to(device)
        #output = model.greedy_decode(vid_inp, input_lens)
        #vid_inp, aud_inp = model.avsr.recon_input(vid_inp, aud_inp, vid_lens, aud_lens)
        output = model.beam_search_decode(vid_inp, aud_inp, vid_lens, aud_lens, bos_id=BOS_ID, eos_id=EOS_ID, max_dec_len=50)
        for out, tgt in zip(output, tgt_txt):
            ## CER
            #preds.append(''.join([test_set.vocab[i] for i in torch.unique_consecutive(out).tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
            preds.append(''.join([test_set.vocab[i] for i in out.tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
            refs.append(''.join([test_set.vocab[i] for i in tgt.tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
            ## WER
            #preds.append(' '.join([test_set.vocab[i] for i in torch.unique_consecutive(out).tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
            #preds.append(' '.join([test_set.vocab[i] for i in out.tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
            #refs.append(' '.join([test_set.vocab[i] for i in tgt.tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]))
            #print(preds[-1], '|||', refs[-1], preds[-1] == refs[-1])
            # write_to('pred-cmlr.txt', ref[-1]+'\t'+pred[-1]+'\t'+str(ref[-1] == pred[-1]))
    test_wer, test_cer = jiwer.wer(refs, preds), jiwer.cer(refs, preds)
    print('JIWER wer: {:.4f}, cer: {:.4f}'.format(test_wer, test_cer))
    return test_wer, test_cer



def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    
    

if __name__ == '__main__':
    set_seed(1337)
    #main()
    evaluate()

