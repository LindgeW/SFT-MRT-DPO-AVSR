import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from multiprocessing import Pool
from conformer import Conformer
from transformer2 import TransformerEncoder
import random
from batch_beam_search import beam_decode


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.se = se

        if self.se:
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.conv3 = conv1x1(planes, planes // 16)
            self.conv4 = conv1x1(planes // 16, planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.se:
            w = self.gap(out)
            w = self.conv3(w)
            w = self.relu(w)
            w = self.conv4(w).sigmoid()
            out = out * w

        out = out + residual
        out = self.relu(out)
        return out


# ResNet18 / ResNet34
class ResNet(nn.Module):
    def __init__(self, block, layers, se=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.se = se
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.bn = nn.BatchNorm1d(512)
         
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, se=self.se))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, se=self.se))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        #x = self.bn(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # PE(pos, 2i)
        pe[:, 1::2] = torch.cos(position * div_term)  # PE(pos, 2i+1)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):  # (B, L, D)
        return self.dropout(x + self.pe[:, :x.size(1)].detach())


class TransDecoder(nn.Module):
    def __init__(self,
                 n_token,
                 d_model,
                 n_layers=3,
                 n_heads=4,
                 ffn_ratio=4,
                 dropout=0.1,
                 max_len=1000):
        super(TransDecoder, self).__init__()
        self.d_model = d_model
        self.scale = d_model**0.5
        self.tok_embedding = nn.Embedding(n_token, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                   dim_feedforward=d_model*ffn_ratio,
                                                   nhead=n_heads,
                                                   dropout=dropout,
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(d_model, n_token)  

    def get_bool_mask_from_lens(self, lengths, max_len=None):
        '''
         param:   lengths --- [Batch_size]
         return:  mask --- [Batch_size, max_len]
        '''
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)   # True for masking
        return mask

    def generate_mask_from_lens(self, seq_lengths, max_length=None):
        """
        根据给定的序列长度生成掩码矩阵。
        Args:
            seq_lengths (torch.Tensor): 每个序列的长度，形状为 (batch_size,)。
            max_length (int): 序列的最大长度。
        Returns:
            torch.Tensor: 掩码矩阵，形状为 (batch_size, max_length)，其中填充部分为 -inf，有效部分为 0。
        """
        if max_length is None:
            max_length = seq_lengths.max().item()
        B = seq_lengths.size(0)
        range_tensor = torch.arange(max_length, device=seq_lengths.device).expand(B, max_length)
        mask = range_tensor < seq_lengths.unsqueeze(1)
        mask = torch.where(mask, 0.0, float('-inf'))
        # mask = mask.float()
        # mask[mask == 0] = float('-inf')
        # mask[mask == 1] = 0.0
        return mask

    def forward(self, tgt, src_enc, src_lens=None, tgt_lens=None):
        tgt = self.pos_enc(self.tok_embedding(tgt) * self.scale)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), dtype=torch.bool).to(tgt.device)   # 下三角 (下0上-inf)
        src_padding_mask = self.get_bool_mask_from_lens(src_lens, src_enc.size(1))   # True for masking
        tgt_padding_mask = self.get_bool_mask_from_lens(tgt_lens, tgt.size(1))   # True for masking
        #src_padding_mask = self.generate_mask_from_lens(src_lens, src_enc.size(1))   # float("-inf") for masking
        #tgt_padding_mask = self.generate_mask_from_lens(tgt_lens, tgt.size(1))   # float("-inf") for masking
        dec_out = self.decoder(tgt, src_enc, tgt_mask,
                               tgt_key_padding_mask=tgt_padding_mask,
                               memory_key_padding_mask=src_padding_mask)
        return self.fc(dec_out)


class AVSRModel(nn.Module):
    def __init__(self, vocab_size, se=False):
        super(AVSRModel, self).__init__()
        
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], se=se)
        #self.resnet34 = ResNet(BasicBlock, [3, 4, 6, 3], se=se)

        self.afront = nn.Sequential(
            nn.Conv1d(80, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),   # downsampling
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),   # downsampling
            nn.BatchNorm1d(512)
        )
        self.scale = 4
        
        #self.am_embed = nn.Parameter(torch.zeros(512))
        #self.vm_embed = nn.Parameter(torch.zeros(512))
       
        #self.av_mlp2 = nn.Sequential(
        #    nn.LayerNorm(512),
        #    nn.Linear(512, 512*2),
        #    nn.ReLU(True),
        #    #nn.GELU(),
        #    nn.Linear(512*2, 512))

        #self.mha_av = nn.MultiheadAttention(embed_dim=512, num_heads=4, dropout=0.1, batch_first=True)
        #self.mha_va = nn.MultiheadAttention(embed_dim=512, num_heads=4, dropout=0.1, batch_first=True)
        #self.av_trans = TransformerEncoder(512, 4, 3, 0.1)

        # backend: gru/tcn/transformer
        #self.gru = nn.GRU(512, 256, num_layers=3, bidirectional=True, batch_first=True, dropout=0.5)
        self.cfm_a = Conformer(512, 4, 512*4, 3, 31, 0.1)
        self.cfm_v = Conformer(512, 4, 512*4, 3, 31, 0.1)
        self.cfm_c = Conformer(512, 4, 512*4, 3, 31, 0.1)

        #self.factor = nn.Parameter(torch.zeros(1))  
        #self.q = nn.Parameter(torch.empty(1, 8, 512).normal_(std=0.02))
        #self.autofusion = AutoFusion(512, 512*2)
        #self.cmd_loss = CMD()
        
        self.ctc_fc = nn.Linear(512, vocab_size)  
        self.trans_dec = TransDecoder(vocab_size, 512, 6, 4) 

        # initialize
        #self._initialize_weights()

    def visual_frontend(self, x):  # (b, c, t, h, w)
        bs = x.size(0)
        x = self.frontend3D(x.transpose(1, 2)).transpose(1, 2).contiguous()
        x = x.reshape(-1, 64, x.size(3), x.size(4))
        x = self.resnet18(x)
        #x = self.resnet34(x)
        return x.reshape(bs, -1, 512)
    
    def audio_frontend(self, x):  # (b, t, c)
        x = self.afront(x.transpose(1, 2)).transpose(1, 2).contiguous()
        return x

    def attention(self, q, k, v):
        s = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        attn = F.softmax(s, dim=-1) @ v
        return attn

    def av_fusion(self, a, v, av_mlp=None):  # (B, Ta, D)  (B, Tv, D)
        #va = v + self.attention(v, a, a) 
        #av = a + self.attention(a, v, v)  
        av = a + self.mha_av(a, v, v)[0]
        #out = av + av_mlp(av)
        out = self.ln(av + av_mlp(av))
        return out

    def av_fusion_memory(self, a, v, av_mlp=None):  # (B, Ta, D)  (B, Tv, D)
        vq = self.attention(self.q, v, v)     # V -->> q
        av = a + self.factor * self.attention(a, vq, vq)    # q -->> A
        #concat_ = torch.cat((a, v), dim=1)
        #avq = self.attention(self.q, concat_, concat_)    # AV -->> q
        #av = a + self.attention(a, avq, avq)    # q -->> AV
        #va = v + self.attention(v, avq, avq)    # q -->> VA
        out = self.ln(av + av_mlp(av))
        return out

    def forward(self, vid, aud, tgt, vid_lens=None, aud_lens=None, tgt_lens=None):  # (b, t, c, h, w)
        enc_src, src_lens = self.encode_av(vid, aud, vid_lens, aud_lens)
        ctc_log_probs = self.ctc_fc(enc_src).log_softmax(dim=-1).transpose(0, 1)  # (T, B, V)
        ctc_loss = F.ctc_loss(ctc_log_probs, tgt[:, 1:], src_lens.reshape(-1), tgt_lens.reshape(-1), zero_infinity=True)  # audio-as-query
        dec_out = self.trans_dec(tgt[:, :-1], enc_src, src_lens=src_lens, tgt_lens=tgt_lens)
        attn_loss = F.cross_entropy(dec_out.transpose(-1, -2).contiguous(), tgt[:, 1:], ignore_index=0)
        loss = 0.9 * attn_loss + 0.1 * ctc_loss
        return loss, enc_src
        
    def encode_av(self, vid, aud, vid_lens=None, aud_lens=None):
        vid_feat = self.visual_frontend(vid)
        aud_feat = self.audio_frontend(aud)
        aud_lens = (aud_lens + self.scale - 1) // self.scale    # time subsampling after CNN striding
        '''
        inp_feat = self.av_fusion(aud, vid)
        enc_src = self.cfm(inp_feat, src_lens)
        '''
        #inp_feat = torch.cat((aud+self.am_embed, vid+self.vm_embed), dim=1)
        #enc_src = self.cfm(inp_feat, src_lens+vid_lens)[:, :aud.shape[1]]
        enc_v, enc_a = self.cfm_v(vid_feat, vid_lens), self.cfm_a(aud_feat, aud_lens)
        #enc_vc, enc_ac = self.cfm_c(vid, vid_lens), self.cfm_c(aud, aud_lens)
        #enc_src = self.av_fusion(enc_ac, enc_vc, self.av_mlp)
        #enc_src = self.av_trans(enc_a.transpose(0, 1), enc_v.transpose(0, 1), enc_v.transpose(0, 1)).transpose(0, 1)
        #enc_src = self.av_trans(enc_vc.transpose(0, 1), enc_ac.transpose(0, 1), enc_ac.transpose(0, 1)).transpose(0, 1)
        enc_src, _ = self.cfm_c.forward_pair(enc_a, aud_lens, enc_v, vid_lens)
        #out2 = self.fc(F.normalize(seq_feat1, dim=-1)) + self.fc(F.normalize(seq_feat2, dim=-1))
        return enc_src, aud_lens
    
    def decoder_forward(self, tgt, enc_memory, src_lens, tgt_lens):
        # tgt: 含bos_id和eos_id
        # 返回 log_probs [Batch, Len, Vocab]
        dec_logits = self.trans_dec(tgt, enc_memory, src_lens=src_lens, tgt_lens=tgt_lens)
        return dec_logits.log_softmax(dim=-1)
    
    def generate(self, enc_memory, src_lens, bos_id, eos_id, max_dec_len=100, beam_size=5):
        # 返回: N-best tokens List[List[Tensor]]
        res = beam_decode(self.trans_dec, enc_memory, src_lens, bos_id, eos_id, max_output_length=max_dec_len, beam_size=beam_size, n_best=beam_size)[0]
        return res
    
    @torch.no_grad()
    def beam_search_decode(self, vid, aud, vid_lens, aud_lens, bos_id, eos_id, max_dec_len=80, pad_id=0):
        enc_src, src_lens = self.encode_av(vid, aud, vid_lens, aud_lens)
        #inp_feat = torch.cat((aud+self.am_embed, vid+self.vm_embed), dim=1)
        #enc_src = self.cfm(inp_feat, lens+vid_lens)[:, :aud.shape[1]]
        #enc_vc, enc_ac = self.cfm_c(vid, vid_lens), self.cfm_c(aud, aud_lens)
        #enc_src = self.av_trans(enc_a.transpose(0, 1), enc_v.transpose(0, 1), enc_v.transpose(0, 1)).transpose(0, 1)
        #enc_src = self.av_trans(enc_vc.transpose(0, 1), enc_ac.transpose(0, 1), enc_ac.transpose(0, 1)).transpose(0, 1)
        #res = beam_decode(self.trans_dec, enc_src, src_mask, bos_id, eos_id, max_output_length=max_dec_len, beam_size=10)
        res = beam_decode(self.trans_dec, enc_src, src_lens, bos_id, eos_id, max_output_length=max_dec_len, beam_size=10).detach().cpu()
        return res

    def ctc_greedy_decode(self, vids, lens=None):
        with torch.no_grad():
            vid_feat = self.visual_frontend(vids)
            seq_feat = self.cfm(vid_feat, lens)
            logits = self.fc(seq_feat)  # (B, T, V)
            return logits.data.cpu().argmax(dim=-1)

    def ctc_beam_decode(self, vids, lens=None):
        res = []
        with torch.no_grad():
            vid_feat = self.visual_frontend(vids)
            seq_feat = self.cfm(vid_feat, lens)
            logits = self.fc(seq_feat)  # (B, T, V)
            probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
            for prob in probs:
                pred = ctc_beam_decode3(prob, 10, 0)
                res.append(pred)
            return res
    
    '''
    def beam_decode(self, vids):
        res = []
        with torch.no_grad():
            logits = self.forward(vids)[0]  # (B, T, V)
            probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
            with Pool(len(probs)) as p:
                res = p.map(ctc_beam_decode3, probs)
                #res.append(pred)
            return res
    '''

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



# 适用于ASR
class SelfAttentivePooling(nn.Module):
    def __init__(self, hid_size, attn_size=None):
        super(SelfAttentivePooling, self).__init__()
        if attn_size is None:
            attn_size = hid_size // 2
        self.mlp = nn.Sequential(
                nn.Linear(hid_size, attn_size),
                nn.ReLU(True),
                nn.Linear(attn_size, 1, bias=False))

    def forward(self, x):  # (B, L, D)
        attn = self.mlp(x).squeeze(2)  # (B, L)
        attn_weights = F.softmax(attn, dim=1)  # (B, L)
        weighted_inputs = torch.mul(x, attn_weights.unsqueeze(2))  # (B, L, D)
        pooled_output = torch.sum(weighted_inputs, dim=1)  # (B, D)
        return pooled_output


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1)
