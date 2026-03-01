import numpy as np
import torch
import random


def horizontal_flip(vid):
    # (T, C, H, W)
    return np.flip(vid, -1).copy()


def vid_dropout(vid, ratio=0.2, p=0.2):   # (T, C, H, W)
    if np.random.uniform() >= p:
        return vid
    T = vid.shape[0]
    rand_probs = np.random.rand(T)
    mask = rand_probs < ratio
    vid_ = vid.copy()
    vid_[mask] = 0
    return vid_


def vid_time_masking(vid, ratio=0.2, p=0.2):   # (T, C, H, W)
    if np.random.uniform() >= p:
        return vid
    T = vid.shape[0]
    L = np.random.randint(0, int(T * ratio))  
    t0 = np.random.randint(0, T - L) 
    mean_frame = vid.mean(0)
    #vid[t0: t0 + L] = 0    
    vid[t0: t0 + L] = mean_frame  
    return vid


def aud_time_masking(wav, ratio=0.4, p=0.2):  # (n_samples, )
    if np.random.uniform() >= p:
        return wav
    N = wav.shape[0]
    L = np.random.randint(0, int(N * ratio))
    t0 = np.random.randint(0, N - L)
    aud = wav.copy()
    aud[t0: t0 + L] = 0.
    return aud


def spec_augment(mel_spec, freq_masking_para=5, time_masking_para=30, freq_mask_num=1, time_mask_num=1, time_first=False):
    """Spec augmentation Calculation Function.
    'specAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      freq_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      freq_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.
    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    if time_first:
        mel_spec = mel_spec.transpose(1, 0)
    # (D, T)
    v = mel_spec.shape[0]
    tau = mel_spec.shape[1]
    # 如果logmel特征做了z-norm处理(均值为0，方差为1)，用0填充相当于均值填充
    repl_val = 0.  # or mel_spec.mean()
    # Step 1: Frequency masking
    for i in range(freq_mask_num):
        f = int(np.random.uniform(low=0.0, high=freq_masking_para))
        f0 = np.random.randint(0, v - f)
        mel_spec[f0:f0 + f, :] = repl_val  
    # Step 2: Time masking
    for i in range(time_mask_num):
        t = int(np.random.uniform(low=0.0, high=time_masking_para))
        t0 = np.random.randint(0, tau - t)
        # t0 = np.random.choice(range(0, tau - t))
        mel_spec[:, t0:t0 + t] = repl_val  
    if time_first:
        mel_spec = mel_spec.transpose(1, 0)
    return mel_spec



def batch_spec_augment(mel_spec, freq_masking_para=10, time_masking_para=30, freq_mask_num=1, time_mask_num=1, time_first=False):
    """Spec augmentation Calculation Function.
    'specAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      freq_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      freq_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.
    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    if time_first: mel_spec = mel_spec.transpose(0, 2, 1)
    # (B, D, T)
    v = mel_spec.shape[1]
    tau = mel_spec.shape[2]
    repl_val = 0.  # or mel_spec.mean()
    
    # Step 1: Frequency masking
    for i in range(freq_mask_num):
        f = int(np.random.uniform(low=0.0, high=freq_masking_para))
        f0 = np.random.randint(0, v - f)
        mel_spec[:, f0:f0 + f, :] = repl_val  
    
    # Step 2: Time masking
    for i in range(time_mask_num):
        t = int(np.random.uniform(low=0.0, high=time_masking_para))
        t0 = np.random.randint(0, tau - t)
        # t0 = np.random.choice(range(0, tau - t))
        mel_spec[:, :, t0:t0 + t] = repl_val
    if time_first: mel_spec = mel_spec.transpose(0, 2, 1)
    return mel_spec



def spec_augment(mel_spec, freq_mask_param=30, time_mask_param=40, num_freq_masks=2, num_time_masks=2):
    """mel_spec: (F, T) tensor"""
    augmented = mel_spec.clone()
    F, T = augmented.shape
    # ----- Frequency Masking -----
    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, max(0, F - f))
        augmented[f0:f0 + f, :] = 0
    # ----- Time Masking -----
    for _ in range(num_time_masks):
        t = random.randint(0, time_mask_param)
        t0 = random.randint(0, max(0, T - t))
        augmented[:, t0:t0 + t] = 0
    return augmented


def spec_aug_batch(mel_batch, freq_mask_param=30, time_mask_param=40):
    """
    mel_batch: (B, F, T)
    """
    out = mel_batch.clone()
    B, F, T = out.size()
    for b in range(B):
        # Frequency mask
        f = torch.randint(0, freq_mask_param, (1,))
        f0 = torch.randint(0, F - f, (1,))
        out[b, f0:f0+f, :] = 0
        # Time mask
        t = torch.randint(0, time_mask_param, (1,))
        t0 = torch.randint(0, T - t, (1,))
        out[b, :, t0:t0+t] = 0
    return out

