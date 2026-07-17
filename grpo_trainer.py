import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
import copy
import math
import random
import numpy as np
import sys
import os
import jiwer
import Levenshtein
from torch.utils.data import DataLoader
#from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from avdataset import GRIDDataset, CMLRDataset, BucketBatchSampler
from avmodel import AVSRModel
from constants import *


def warmup_constant_schedule(optimizer, warmup_steps=500):
    """
    前warmup_steps: 0 → 线性升到max_lr
    之后: 保持最大学习率
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0  # 保持最大lr

    return LambdaLR(optimizer, lr_lambda)


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    min_lr_ratio=0.1,
):
    """
    Linear Warmup + Cosine Decay

    min_lr_ratio:
        最终学习率 = peak_lr * min_lr_ratio
    """
    assert 0 <= min_lr_ratio <= 1.0, "min_lr_ratio must between 0 and 1"
    assert num_training_steps > 0, "total training steps must > 0"
    assert num_warmup_steps >= 0, "warmup steps cannot be negative"

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = (current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)



class JointLoss(nn.Module):
    def __init__(self, mrt_alpha=0.05, grpo_eps=0.2, grpo_beta=0.01):
        super().__init__()
        self.mrt_alpha = mrt_alpha
        # GRPO 超参数
        self.grpo_eps = grpo_eps  # PPO 裁剪参数
        #self.grpo_eps_low, self.grpo_eps_high = 0.2, 0.28  # DAPO 裁剪参数
        #self.grpo_eps_low, self.grpo_eps_high = 0.16, 0.24  
        self.grpo_beta = grpo_beta   # KL 惩罚系数 0.01 - 0.05

        self.delta = 2   # 1.5, 2., 2.5

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

    def compute_grpo_loss(self, policy_logps, old_policy_logps, ref_logps, rewards, mask):
        """
        标准 GRPO (Group Relative Policy Optimization) 实现
        
        基于 DeepSeek-R1 论文公式:
        J_GRPO = E[ (1/G) * sum( min(ratio * A, clip(ratio) * A) - beta * KL(pi||pi_ref) ) ]
        
        其中:
        - ratio = pi_theta / pi_old (这里 pi_old 就是采样时的策略，在GRPO中通常直接近似为当前策略的一次前向)
        - A = (reward - mean) / std (组内标准化优势)
        - KL 使用 Schulman 提出的无偏估计: ref/policy - log(ref/policy) - 1
        
        Args:
            policy_logps: [Batch*Group_Size, Seq_len] 当前策略的 log probs
            ref_logps: [Batch*Group_Size, Seq_len] 参考模型的 log probs
            rewards: [Batch, Group_Size] 奖励值 (例如 -WER)
            mask: [Batch*Group_Size, Seq_len] 有效token掩码 1=valid, 0=pad

        Returns:
            loss: GRPO 损失
            metrics: 包含 KL 散度等信息的字典
        """
        batch_size, group_size = rewards.shape
        
        # 计算概率比率 (重要性采样比率)
        # 注意: 在标准GRPO中，采样来自旧策略，但通常我们假设策略变化缓慢，
        # 可以直接使用当前策略计算，或者使用存储的旧策略概率
        # 这里简化为直接使用当前策略（单次更新版本）
        
        # 计算组内相对优势 (Group Relative Advantage)
        # GRPO核心: 不需要critic，直接使用组内标准化奖励作为优势
        #mean_rewards = rewards.mean(dim=-1, keepdim=True)  # [Batch, 1]
        #mean_rewards = torch.cat([rewards, torch.full((batch_size, 1), 1., device=rewards.device)], dim=-1).mean(dim=-1, keepdim=True)  # [Batch, 1]  NGRPO
        std_rewards, mean_rewards = torch.std_mean(torch.cat([rewards, torch.full((batch_size, 1), 1., device=rewards.device)], dim=-1), dim=-1, keepdim=True)  # [Batch, 1]  NGRPO
        #std_rewards = rewards.std(dim=-1, keepdim=True)   # [Batch, 1]
        advantages = (rewards - mean_rewards) / (std_rewards + 1e-8)   # [Batch, Group_Size]
        #advantages = rewards - mean_rewards   # Dr. GRPO  [Batch, Group_Size]
        advantages = advantages.view(-1, 1)   # (B*G, 1) 
        
        # 计算概率比率 (这里简化为1，因为GRPO通常是在采样后立即更新)
        # 如果需要多轮更新，应该存储采样时的log probs作为old_logps
        #ratio = torch.exp(policy_logps - old_policy_logps.detach())  # (B*G, T) 
        #ratio = (policy_logps - old_policy_logps.detach()).clamp(min=-10., max=10.).exp()  
        ratio = (policy_logps - old_policy_logps.detach()).clamp(min=-5., max=5.).exp()  
   
        # PPO 裁剪目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.grpo_eps, 1 + self.grpo_eps) * advantages
        ##surr2 = torch.clamp(ratio, 1 - self.grpo_eps_low, 1 + self.grpo_eps_high) * advantages
        #policy_loss = -torch.min(surr1, surr2)
        #policy_loss = -torch.where(advantages < 0, torch.max(torch.min(surr1, surr2), 3. * advantages), torch.min(surr1, surr2))   # dual-clip
        #policy_loss = -torch.max(torch.min(surr1, surr2), torch.clamp(ratio, 1. / self.delta, self.delta) * advantages)   # double-sided-clip
        policy_loss = -torch.min(torch.min(ratio, torch.tensor(3., device=ratio.device)) * advantages, surr2)
        '''
        taus = torch.where(advantages > 0, 1.0, 1.05)
        gates = torch.sigmoid(taus * (ratio - 1.)) * (4. / taus)
        policy_loss = -gates * advantages
        '''

        if self.grpo_beta > 0:
            # KL 惩罚项 (Reverse KL: KL(ref||policy))
            # 使用 Schulman 提出的无偏估计器 (k3)  KL = ref/policy - log(ref/policy) - 1
            #log_ratio = (ref_logps - policy_logps).clamp(-10, 10)  # log(ref/policy)
            #kl_penalty = (log_ratio.exp() - log_ratio - 1.).clamp(-10, 10)   # Reverse KL 估计: KL(ref||policy)
            kl_penalty = 0.5 * (policy_logps - ref_logps).pow(2)   # k2
            total_loss = policy_loss + self.grpo_beta * kl_penalty
        else:
            total_loss = policy_loss 

        #total_loss = (total_loss * mask).sum() / mask.sum().clamp(min=1.)   # DAPO / verl框架
        total_loss = ((total_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.)).mean()   # GRPO
        #total_loss = ((total_loss * mask).sum(dim=-1) / 50).mean()   # Dr. GRPO (C=50)

        metrics = {
            'policy_loss': total_loss.item(),
            #'kl_penalty': kl_penalty.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_ratio': ratio.mean().item(),
            'max_ratio': ratio.max().item(),
            'mean_reward': rewards.mean().item(),
            #'std_reward': std_rewards.mean().item()
        }
        
        return total_loss, metrics


    def compute_gspo_loss(self, policy_logps, old_policy_logps, ref_logps, rewards, mask):
        """
        标准 GSPO (Group Sequence Policy Optimization) 实现
        
        基于 DeepSeek-R1 论文公式:
        J_GRPO = E[ (1/G) * sum( min(ratio * A, clip(ratio) * A) - beta * KL(pi||pi_ref) ) ]
        
        其中:
        - ratio = pi_theta / pi_old (这里 pi_old 就是采样时的策略，在GRPO中通常直接近似为当前策略的一次前向)
        - A = (reward - mean) / std (组内标准化优势)
        - KL 使用 Schulman 提出的无偏估计: ref/policy - log(ref/policy) - 1
        
        Args:
            policy_logps: [Batch*Group_Size, Seq_len] 当前策略的 log probs
            ref_logps: [Batch*Group_Size, Seq_len] 参考模型的 log probs
            rewards: [Batch, Group_Size] 奖励值 (例如 -WER)
            mask: [Batch*Group_Size, Seq_len] 有效token掩码 1=valid, 0=pad

        Returns:
            loss: GRPO 损失
            metrics: 包含 KL 散度等信息的字典
        """
        batch_size, group_size = rewards.shape
        
        seq_lens = mask.sum(dim=-1).clamp(min=1.)
        
        # 计算概率比率 (重要性采样比率)
        # 注意: 在标准GRPO中，采样来自旧策略，但通常我们假设策略变化缓慢，
        # 可以直接使用当前策略计算，或者使用存储的旧策略概率
        # 这里简化为直接使用当前策略（单次更新版本）
        
        # 计算组内相对优势 (Group Relative Advantage)
        # GRPO核心: 不需要critic，直接使用组内标准化奖励作为优势
        mean_rewards = rewards.mean(dim=-1, keepdim=True)  # [Batch, 1]
        std_rewards = rewards.std(dim=-1, keepdim=True).clamp(min=1e-6)  # [Batch, 1]
        advantages = (rewards - mean_rewards) / std_rewards  # [Batch, Group_Size]
        advantages = advantages.view(-1)   # (B*G, )
        
        # 计算概率比率 (这里简化为1，因为GRPO通常是在采样后立即更新)
        # 如果需要多轮更新，应该存储采样时的log probs作为old_logps
        logps_diff = policy_logps - old_policy_logps.detach()
        seq_logps = (logps_diff * mask).sum(dim=-1) / seq_lens
        ratio = torch.exp(seq_logps)  # (B*G, ) 
        
        # PPO 裁剪目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.grpo_eps, 1 + self.grpo_eps) * advantages
        #surr2 = torch.clamp(ratio, 1 - self.grpo_eps_low, 1 + self.grpo_eps_high) * advantages
        policy_loss = -torch.min(surr1, surr2)
       
        if self.grpo_beta > 0:
            # KL 惩罚项 (Reverse KL: KL(ref||policy))
            # 使用 Schulman 提出的无偏估计器 (k3)  KL = ref/policy - log(ref/policy) - 1
            #log_ratio = ref_logps - policy_logps  # log(ref/policy)
            #seq_log_ratio = (log_ratio * mask).sum(dim=-1) / seq_lens
            #seq_kl = (seq_log_ratio.exp() - seq_log_ratio - 1.)  # Reverse KL 估计: KL(ref||policy)
            
            #seq_kl = ((policy_logps - ref_logps) * mask).sum(dim=-1) / seq_lens  # k1
            seq_kl = (0.5 * (policy_logps - ref_logps).pow(2) * mask).sum(dim=-1) / seq_lens  # k2
            
            total_loss = (policy_loss + self.grpo_beta * seq_kl).mean()
        else:
            total_loss = policy_loss.mean()
       
        metrics = {
            'policy_loss': total_loss.item(),
            #'kl_penalty': kl_penalty.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_ratio': ratio.mean().item(),
            'max_ratio': ratio.max().item(),
            'mean_reward': rewards.mean().item(),
            'std_reward': std_rewards.mean().item()
        }
        
        return total_loss, metrics



class BaseTrainer:
    def __init__(self, model, optimizer, lr_scheduler=None, accumulate_step=1, device='cuda:0'):
        self.model = model
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
    def __init__(self, model, optimizer, tokenizer, lambda_ce=0.2, lambda_mrt=1.0, lambda_grpo=0.1, device='cuda:0', grpo_beta=0.01, grpo_group_size=8):
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer   # 用于计算 WER 时的解码
        self.pad_id, self.bos_id, self.eos_id = (tokenizer.index(x) for x in [PAD, BOS, EOS])
        print(f'PAD ID: {self.pad_id}, BOS ID: {self.bos_id}, EOS ID: {self.eos_id}')
        self.device = device
        
        # GRPO 需要参考模型（ref_model）来计算 KL 惩罚
        # 参考模型是固定的初始SFT模型，不更新
        #self.ref_model = copy.deepcopy(model)
        #for param in self.ref_model.parameters():
        #    param.requires_grad = False
        #self.ref_model.eval()
        self.ref_model = None
        
        self.criterion = JointLoss(mrt_alpha=1., grpo_eps=0.2, grpo_beta=grpo_beta)
        self.weights = {'ce': lambda_ce, 'mrt': lambda_mrt, 'grpo': lambda_grpo}
        
        # GRPO 组大小 (每个样本采样的候选数量)
        self.grpo_group_size = grpo_group_size

    def get_batch_log_probs(self, model, tgt_tokens, enc_memory, src_lens, tgt_lens, len_norm=False):
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
            reward = -wer
            rewards.append(reward)
        return rewards

    def train_step(self, vid_inp, aud_inp, tgt_tokens, vid_lens, aud_lens, tgt_lens, n_best=5):
        """
        vid_inp: [B, C, T, H, W]
        aud_inp: [B, T_a, D]
        tgt_tokens: [B, L]
        """
        batch_size = vid_inp.size(0)
        
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
                    txt = ''.join([self.tokenizer[i] for i in pred.tolist() if i not in [self.pad_id, self.bos_id, self.eos_id]]).strip()
                    batch_out.append({'text': txt, 'tokens': torch.cat([torch.tensor([self.bos_id], device=pred.device), pred])})
                nbest_batch.append(batch_out)
                
                tgt_text.append(''.join([self.tokenizer[i] for i in tgts.tolist() if i not in [self.pad_id, self.bos_id, self.eos_id]]).strip())
            
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
                    rewards.append(-wer)  # 奖励 = -WER
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
        # Step 3: 当前策略 (Policy) 计算 Log-Probs (With Grad)
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

        # 单次更新
        old_policy_logps = policy_grpo_logps.detach()

        # -------------------------------------------
        # Step 4: 参考模型 (Reference) 计算 Log-Probs (No Grad)
        # -------------------------------------------
        with torch.no_grad():
            # 参考模型用于计算 KL 惩罚
            ref_enc_memory, ref_src_lens = self.ref_model.encode_av(vid_inp, aud_inp, vid_lens, aud_lens)
            ref_grpo_mem_expanded = ref_enc_memory.repeat_interleave(group_size, dim=0)
            ref_grpo_src_lens_expanded = ref_src_lens.repeat_interleave(group_size, dim=0)
            
            ref_grpo_logps = self.get_batch_log_probs(self.ref_model, grpo_inputs, ref_grpo_mem_expanded, ref_grpo_src_lens_expanded, grpo_tgt_lens_expanded)
            ref_grpo_logps = ref_grpo_logps.view(batch_size, group_size)

        # -------------------------------------------
        # Step 5: 计算联合损失
        # -------------------------------------------
        # 1. MRT Loss
        loss_mrt = self.criterion.compute_mrt_loss(all_mrt_logps_view, mrt_risks)
        
        # 2. GRPO Loss
        loss_grpo, grpo_metrics = self.criterion.compute_grpo_loss(
            policy_grpo_logps, old_policy_logps, ref_grpo_logps, grpo_rewards
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
        
        #nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()

        return {
            "total": total_loss.item(),
            "mrt": loss_mrt.item(),
            "grpo": loss_grpo.item(),
            "ce": loss_ce.item(),
            "grpo_policy_loss": grpo_metrics['policy_loss'],
            "grpo_kl": grpo_metrics['kl_penalty'],
            "grpo_mean_adv": grpo_metrics['mean_advantage']
        }
        


class MRT_GRPO_Trainer_MultiStep:
    def __init__(self, model, optimizer, tokenizer, lambda_ce=0.2, lambda_mrt=1.0, lambda_grpo=0.1, device='cuda:0', grpo_group_size=8, grpo_beta=0.01, num_updates=4):
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.pad_id, self.bos_id, self.eos_id = (tokenizer.index(x) for x in [PAD, BOS, EOS])
        self.device = device
        self.grpo_beta = grpo_beta

        # 参考模型
        self.ref_model = copy.deepcopy(model)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        self.criterion = JointLoss(mrt_alpha=1., grpo_eps=0.2, grpo_beta=grpo_beta)
        self.weights = {'ce': lambda_ce, 'mrt': lambda_mrt, 'grpo': lambda_grpo}
        #self.weights = {'ce': 0., 'mrt': 0., 'grpo': 1.}
        self.grpo_group_size = grpo_group_size
        self.num_updates = num_updates  # 多步更新次数 PPO: 4-10, GRPO: 1-4

        # GRPO缓冲区
        self.grpo_buffer = None
    
    def get_batch_log_probs(self, model, tgt_tokens, enc_memory, src_lens, tgt_lens, len_norm=False):
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

    def get_token_log_probs(self, model, tgt_tokens, enc_memory, src_lens, tgt_lens):
        """
            计算给定 Token 序列在模型下的 Log-Prob 
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
        
        target_log_probs = torch.gather(preds_log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # (B, L)
        
        # Mask padding (假设 pad_id = 0)
        pad_mask = (targets != 0).float()
             
        return target_log_probs, pad_mask

    def update_old_policy(self):
        """
        更新旧策略或参考模型（用于 GRPO 的重要性采样）
        可以选择在每个 epoch 开始时更新，或每隔几步更新
        """
        # 深拷贝当前模型作为旧策略
        # self.old_model = copy.deepcopy(self.model)
        self.ref_model = copy.deepcopy(self.model)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        # self.old_model.eval()
        self.ref_model.eval()

    def type_aware_wer(self, ref_str, hyp_str, w_s=1., w_d=1., w_i=1.):
        """
        计算类型感知的加权 WER，作为 MRT 的 Risk 或 DPO 的负 Reward。
        w_s: 替换错误 (Substitution) 的权重，AVSR 中建议调高，打击嘴型歧义猜词。
        w_d: 删除错误 (Deletion) 的权重。
        w_i: 插入错误 (Insertion) 的权重。
        """
        # 边界情况处理：如果两者都为空，风险为 0
        if not ref_str.strip() and not hyp_str.strip():
            return 0.0
        # 边界情况处理：如果 ref 为空，hyp 不为空，全算作 Insertion
        if not ref_str.strip():
            return w_i * len(hyp_str.split())
        
        # 使用 jiwer 直接获取精确的 S, D, I 统计
        measures = jiwer.compute_measures(ref_str, hyp_str)
        S = measures['substitutions']
        D = measures['deletions']
        I = measures['insertions']
        N = measures['hits'] + S + D  # 严格等于真实文本的单词数
       
        return (w_s * S + w_d * D + w_i * I) / max(N, 1)
        #L = (len(ref_str.split()) + len(hyp_str.split())) / 2.
        #return (w_s * S + w_d * D + w_i * I) / max(L, 1)
   
    def token_accuracy(self, ref_tokens, hyp_tokens):
        # 使用编辑距离回溯对齐（简化示例，实际可用 python-Levenshtein.opcodes）
        correct = 0
        ops = Levenshtein.opcodes(ref_tokens, hyp_tokens)
        for op, i1, i2, j1, j2 in ops:
            if op == 'equal':
                correct += (i2 - i1)
        return correct / max(len(ref_tokens), 1)
        #return 2 * correct / (len(ref_tokens) + len(hyp_tokens))   # F1

    @torch.no_grad()
    def sample_and_store(self, vid_inp, aud_inp, tgt_tokens, vid_lens, aud_lens, tgt_lens, n_best=5):
        """
        采样阶段：生成候选并存储到buffer（无梯度）
        """
        batch_size = vid_inp.size(0)

        with torch.no_grad():
            # 计算encoder用于生成
            enc_memory, src_lens = self.model.encode_av(vid_inp, aud_inp, vid_lens, aud_lens)

            # 生成N-best
            nbest_output = self.model.generate(
                enc_memory, src_lens, self.bos_id, self.eos_id,
                max_dec_len=50, beam_size=n_best
            )

            # 收集数据
            grpo_tokens_list = []
            grpo_rewards_list = []
            mrt_tokens_list = []
            mrt_risks_list = []

            # 解析目标文本
            tgt_text = []
            for tgts in tgt_tokens:
                tgt_text.append(''.join([
                    self.tokenizer[i] for i in tgts.tolist()
                    if i not in [self.pad_id, self.bos_id, self.eos_id]
                ]).strip())

            for b in range(batch_size):
                hyps = nbest_output[b]
                ref_text = tgt_text[b]

                wers = []
                rewards = []
                current_batch_tokens = []

                for pred in hyps[:n_best]:
                    hyp_txt = ''.join([
                        self.tokenizer[i] for i in pred.tolist()
                        if i not in [self.pad_id, self.bos_id, self.eos_id]
                    ]).strip()

                    wer = jiwer.wer(ref_text, hyp_txt) + jiwer.cer(ref_text, hyp_txt)   # 8.13 
                    #wer = self.type_aware_wer(ref_text, hyp_txt, 2., 1.5, 1.5)   # 8.25
                    wers.append(wer)

                    rwd = np.exp(-3. * wer) # (0, 1]
                    #rwd = 1. - min(wer, 2.0) # [-1, 1]
                    #rwd = 1. - min(wer, 2.0) + self.token_accuracy(ref_text.split(), hyp_txt.split())
                    rewards.append(rwd)
                    current_batch_tokens.append(
                        torch.cat([torch.tensor([self.bos_id], device=pred.device), pred])
                    )
                             
                group_size = min(self.grpo_group_size, len(current_batch_tokens))
                selected_indices = list(range(group_size))

                grpo_tokens_list.extend([current_batch_tokens[i] for i in selected_indices])
                grpo_rewards_list.extend([rewards[i] for i in selected_indices])

                current_batch_tokens.append(tgt_tokens[b])
                wers.append(0.0)
                mrt_tokens_list.extend(current_batch_tokens)
                mrt_risks_list.extend(wers)

            grpo_inputs = pad_sequence(grpo_tokens_list, batch_first=True, padding_value=self.pad_id).to(self.device)
            grpo_rewards = torch.tensor(grpo_rewards_list).view(batch_size, group_size).to(self.device)
            mrt_inputs = pad_sequence(mrt_tokens_list, batch_first=True, padding_value=self.pad_id).to(self.device)
            mrt_risks = torch.tensor(mrt_risks_list).view(batch_size, n_best + 1).to(self.device)

            # 计算old_policy_logps（采样时的策略）
            old_enc_expanded = enc_memory.repeat_interleave(group_size, dim=0)
            old_src_lens_expanded = src_lens.repeat_interleave(group_size, dim=0)
            old_tgt_lens_expanded = tgt_lens.repeat_interleave(group_size, dim=0)

            old_policy_logps, old_pad_mask = self.get_token_log_probs(
                self.model, grpo_inputs, old_enc_expanded,
                old_src_lens_expanded, old_tgt_lens_expanded
            )   # (B*G, T)

            # 存储到buffer
            self.grpo_buffer = {
                'batch_size': batch_size,
                'group_size': group_size,
                'n_best': n_best,
                'vid_inp': vid_inp,
                'aud_inp': aud_inp,
                'tgt_tokens': tgt_tokens,
                'vid_lens': vid_lens,
                'aud_lens': aud_lens,
                'tgt_lens': tgt_lens,
                'grpo_inputs': grpo_inputs,
                'grpo_rewards': grpo_rewards,
                'mrt_inputs': mrt_inputs,
                'mrt_risks': mrt_risks,
                'old_policy_logps': old_policy_logps.detach(),  # 关键：固定不变！
                'old_pad_mask': old_pad_mask
            }

        return True

 
    def grpo_update_step(self, update_idx):
        """
        单步更新：使用buffer中的数据计算loss并更新
        """
        buffer = self.grpo_buffer
        if buffer is None:
            raise ValueError("GRPO buffer is empty!")

        batch_size = buffer['batch_size']
        group_size = buffer['group_size']
        n_best = buffer['n_best']

        # 重新计算当前策略（有梯度）
        enc_memory, src_lens = self.model.encode_av(
            buffer['vid_inp'], buffer['aud_inp'],
            buffer['vid_lens'], buffer['aud_lens']
        )

        # GRPO部分
        enc_expanded = enc_memory.repeat_interleave(group_size, dim=0)
        src_lens_expanded = src_lens.repeat_interleave(group_size, dim=0)
        tgt_lens_expanded = buffer['tgt_lens'].repeat_interleave(group_size, dim=0)

        policy_grpo_logps, pad_mask = self.get_token_log_probs(
            self.model, buffer['grpo_inputs'], enc_expanded,
            src_lens_expanded, tgt_lens_expanded
        )

        if self.ref_model is not None and self.grpo_beta > 0:
            # 参考模型
            with torch.no_grad():
                ref_enc_memory, ref_src_lens = self.ref_model.encode_av(
                    buffer['vid_inp'], buffer['aud_inp'],
                    buffer['vid_lens'], buffer['aud_lens']
                )
                ref_enc_expanded = ref_enc_memory.repeat_interleave(group_size, dim=0)
                ref_grpo_logps, _ = self.get_token_log_probs(
                    self.ref_model, buffer['grpo_inputs'], ref_enc_expanded,
                    ref_src_lens.repeat_interleave(group_size, dim=0), tgt_lens_expanded
                )
        else:
            ref_grpo_logps = None
        
        loss_grpo, grpo_metrics = self.criterion.compute_grpo_loss(
            policy_grpo_logps,
            buffer['old_policy_logps'],  # 固定baseline
            ref_grpo_logps,
            buffer['grpo_rewards'],
            pad_mask
        )
        
        # 新增：GT监督约束(CE loss)，使用当前正在更新的模型，计算 Ground Truth 的概率
        #gt_logps = self.get_batch_log_probs(
        #    self.model,
        #    buffer['tgt_tokens'],
        #    enc_memory,           # 复用刚刚算好的当前 encoder 输出
        #    src_lens,
        #    buffer['tgt_lens'],
        #    len_norm=True         # 开启长度归一化，等价于 Token 级的平均 CE Loss
        #)
        #loss_ce = -gt_logps.mean()

        total_loss = self.weights['grpo'] * loss_grpo
        #total_loss = self.weights['grpo'] * loss_grpo + self.weights['ce'] * loss_ce

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Update {update_idx}: Invalid loss, skipping")
            return None

        self.optimizer.zero_grad()

        total_loss.backward()
        
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.)
        
        self.optimizer.step()

        return {
            "total": total_loss.item(),
            "grpo": loss_grpo.item(),
            'update_idx': update_idx,
            'mean_ratio': grpo_metrics['mean_ratio'],
            'max_ratio': grpo_metrics['max_ratio'],
        }
        
        
    def mrt_update_step(self):
        """
        单步MRT+CE更新：使用buffer中的数据计算loss并更新
        """
        buffer = self.grpo_buffer
        if buffer is None:
            raise ValueError("GRPO buffer is empty!")

        batch_size = buffer['batch_size']
        n_best = buffer['n_best']

        # 重新计算当前策略（有梯度）
        enc_memory, src_lens = self.model.encode_av(
            buffer['vid_inp'], buffer['aud_inp'],
            buffer['vid_lens'], buffer['aud_lens']
        )

        # MRT部分
        mrt_enc_expanded = enc_memory.repeat_interleave(n_best + 1, dim=0)

        all_mrt_logps = self.get_batch_log_probs(
            self.model, buffer['mrt_inputs'], mrt_enc_expanded,
            #src_lens.repeat_interleave(n_best + 1, dim=0), tgt_lens_expanded
            src_lens.repeat_interleave(n_best + 1, dim=0), buffer['tgt_lens'].repeat_interleave(n_best + 1, dim=0)
        )
        all_mrt_logps_view = all_mrt_logps.view(batch_size, n_best + 1)

        # 计算损失
        loss_mrt = self.criterion.compute_mrt_loss(all_mrt_logps_view, buffer['mrt_risks'])
        
        loss_ce = -all_mrt_logps_view[:, -1].mean()   # GT log_probs

        total_loss = self.weights['mrt'] * loss_mrt + self.weights['ce'] * loss_ce

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("Invalid loss, skipping")
            return None

        self.optimizer.zero_grad()

        total_loss.backward()
        
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.)
        
        self.optimizer.step()

        return {
            "total": total_loss.item(),
            "mrt": loss_mrt.item(),
            "ce": loss_ce.item()
        }
        

    def train_step(self, vid_inp, aud_inp, tgt_tokens, vid_lens, aud_lens, tgt_lens):
        """
        完整训练步骤：采样 + 多步更新
        """
        # 阶段1：采样并存储
        self.sample_and_store(vid_inp, aud_inp, tgt_tokens, vid_lens, aud_lens, tgt_lens, n_best=self.grpo_group_size)

        # 阶段2：多步GRPO更新
        grpo_results = []
        for update_idx in range(self.num_updates):
            result = self.grpo_update_step(update_idx)
            if result is not None:
                grpo_results.append(result)
                #print(f"Update {update_idx}: loss={result['total']:.4f}, ratio={result['mean_ratio']:.4f}")

        #mrt_res = self.mrt_update_step()  # MRT + CE更新
        mrt_res = {'ce': 0., 'mrt': 0.}

        # 返回最后一步的结果
        last_res = grpo_results[-1]
        #return last_res if grpo_results else {'total': 0.0, 'skipped': True}
        return {
            'total': last_res['total'],
            'grpo': last_res['grpo'],
            'mrt': mrt_res['mrt'],
            'ce': mrt_res['ce'],
            'mean_ratio': last_res['mean_ratio'],
            'max_ratio': last_res['max_ratio'],
        }



def get_grouped_optimizer(model, base_lr=2e-6, encoder_lr_ratio=0.1):
    """
    获取分层学习率的优化器
    base_lr: Decoder (策略核心) 的基础学习率
    encoder_lr_ratio: Encoder (特征提取) 的学习率缩放比例 (比如 0.1 表示 2e-7)
    """
    decoder_params = []
    encoder_params = []
    dec_param_count = 0
    enc_param_count = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 只要名字里包含 "trans_dec"，就归给 Decoder
        # (假设你的词表映射层 FC 也需要较大的 LR，如果 FC 不在 trans_dec 里，请用 or 加上它)
        if "trans_dec" in name or "fc" in name: 
            decoder_params.append(param)
            dec_param_count += param.numel()
        else:
            # 剩下的所有烂七八糟名字的模块 (3DCNN, 1DCNN, Conformer等) 全归 Encoder
            encoder_params.append(param)
            enc_param_count += param.numel()

    # 打印参数分布，确保没有分错
    print(f"[Optimizer Setup] Decoder params (LR={base_lr}): {dec_param_count / 1e6:.2f} M")
    print(f"[Optimizer Setup] Encoder params (LR={base_lr * encoder_lr_ratio}): {enc_param_count / 1e6:.2f} M")

    optimizer_grouped_parameters = [
        {'params': encoder_params, 'lr': base_lr * encoder_lr_ratio},
        {'params': decoder_params, 'lr': base_lr}
    ]

    optimizer = optim.AdamW(
        optimizer_grouped_parameters, 
        betas=(0.9, 0.98), 
        eps=1e-9, 
        weight_decay=1e-4
    )

    return optimizer


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
    model.train()
    print(model)

    #model.frozen_encoder()

    batch_size = 32
    accumulate_step = 1
    epochs = {'I': 50, 'II': 5}   
    #lrs = {'I': 3e-4, 'II': 5e-6}  # 7.91 
    lrs = {'I': 3e-4, 'II': 1e-5}  # 7.84 
    savedir = os.path.join('checkpoints', 'grid')

    
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


    train_set.noise_ratio = 0.
    data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=train_set.collate_pad)  
    optimizer2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lrs['II'], betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
    #optimizer2 = get_grouped_optimizer(model, base_lr=lrs['II'], encoder_lr_ratio=0.1)
    ## II MRT-GRPO (替换原来的 MRT-DPO)
    mrt_grpo_trainer = MRT_GRPO_Trainer_MultiStep(model, optimizer2, train_set.vocab, lambda_ce=0.2, lambda_mrt=0.8, lambda_grpo=1., device=device, grpo_group_size=6, grpo_beta=0.01, num_updates=2)  # 8.25
    num_iters = len(data_loader) * epochs['II']
    lr_scheduler2 = get_cosine_schedule_with_warmup(optimizer2, num_warmup_steps=num_iters // 10, num_training_steps=num_iters, min_lr_ratio=0.1)   
    #lr_scheduler2 = warmup_constant_schedule(optimizer2, warmup_steps=num_iters // 10)
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
            lr_scheduler2.step()
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
            #preds.append(''.join([test_set.vocab[i] for i in torch.unique_consecutive(out).tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]).strip())
            preds.append(''.join([test_set.vocab[i] for i in out.tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]).strip())
            refs.append(''.join([test_set.vocab[i] for i in tgt.tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]).strip())
            ## WER
            #preds.append(' '.join([test_set.vocab[i] for i in torch.unique_consecutive(out).tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]).strip())
            #preds.append(' '.join([test_set.vocab[i] for i in out.tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]).strip())
            #refs.append(' '.join([test_set.vocab[i] for i in tgt.tolist() if i not in [PAD_ID, BOS_ID, EOS_ID]]).strip())
            #print('pred:ref', preds[-1], '||', refs[-1], '||', preds[-1] == refs[-1])
            #write_to('grpo_pred2.txt', refs[-1]+'\t'+preds[-1]+'\t'+str(refs[-1] == preds[-1]))
            #write_to('sft_pred.txt', refs[-1]+'\t'+preds[-1]+'\t'+str(refs[-1] == preds[-1]))
    test_wer, test_cer = jiwer.wer(refs, preds), jiwer.cer(refs, preds)
    print('JIWER wer: {:.4f}, cer: {:.4f}'.format(test_wer, test_cer))
    return test_wer, test_cer


def write_to(fn, line):
    with open(fn, 'a', encoding='utf-8') as fw:
        fw.write(line.strip() + '\n')


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
    main()
    # evaluate()

