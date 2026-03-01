import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import json
import librosa
from torch.nn.utils.rnn import pad_sequence
import string
from skimage.util import random_noise
import torchvision
from data_augment import *
from constants import *


# class BucketBatchSampler(Sampler):
#     '''
#         确保每个batch中的样本长度相似，减少padding数量，更高效地利用显存，提高训练效率
#     '''
#     def __init__(self, lens, batch_size, bucket_boundaries):
#         self.batch_size = batch_size
#         self.bucket_boundaries = bucket_boundaries
#         self.num_buckets = len(bucket_boundaries) + 1   # 桶数
#         self.bucket_sizes = [0] * self.num_buckets    # 每个桶的大小
#         self.bucket_indices = {i: [] for i in range(self.num_buckets)}
#
#         # 将数据分配到不同的桶中
#         for idx, length in enumerate(lens):
#             bucket_idx = self._get_bucket_index(length)
#             self.bucket_indices[bucket_idx].append(idx)
#             self.bucket_sizes[bucket_idx] += 1
#
#     def _get_bucket_index(self, length):
#         for i, boundary in enumerate(self.bucket_boundaries):
#             if length <= boundary:
#                 return i
#         return len(self.bucket_boundaries)
#
#     def __iter__(self):
#         # 打乱每个桶中的数据
#         for bucket_idx in range(self.num_buckets):
#             np.random.shuffle(self.bucket_indices[bucket_idx])
#         _batch = []
#         for bucket_idx in range(self.num_buckets):
#             for idx in self.bucket_indices[bucket_idx]:
#                 _batch.append(idx)
#                 if len(_batch) == self.batch_size:
#                     yield _batch
#                     _batch = []
#         if _batch:
#             yield _batch
#
#     def __len__(self):
#         return sum(self.bucket_sizes)


class BucketBatchSampler(Sampler):
    '''
        确保每个batch中的样本长度相似，减少padding数量，更高效地利用显存，提高训练效率
    '''
    def __init__(self, dataset, batch_size, bucket_boundaries):
        self.batch_size = batch_size
        self.bucket_boundaries = bucket_boundaries
        self.num_buckets = len(bucket_boundaries) + 1   # 桶数
        self.bucket_sizes = [0] * self.num_buckets    # 每个桶的大小
        self.bucket_indices = {i: [] for i in range(self.num_buckets)}

        # 将数据分配到不同的桶中
        for idx, item in enumerate(dataset):
            length = len(item['vid'])   # 根据视频帧长进行分桶（这种取data长度的方式速度比较慢，最好直接传长度）
            bucket_idx = self._get_bucket_index(length)
            self.bucket_indices[bucket_idx].append(idx)
            self.bucket_sizes[bucket_idx] += 1

    def _get_bucket_index(self, length):
        for i, boundary in enumerate(self.bucket_boundaries):
            if length <= boundary:
                return i
        return len(self.bucket_boundaries)

    def __iter__(self):
        # 打乱每个桶中的数据
        for bucket_idx in range(self.num_buckets):
            np.random.shuffle(self.bucket_indices[bucket_idx])
        _batch = []
        for bucket_idx in range(self.num_buckets):
            for idx in self.bucket_indices[bucket_idx]:
                _batch.append(idx)
                if len(_batch) == self.batch_size:
                    yield _batch
                    _batch = []
        if _batch:
            yield _batch

    def __len__(self):
        return sum(self.bucket_sizes)


# class BucketBatchSampler(Sampler):
#     def __init__(self, dataset, batch_size, bucket_size, shuffle=True, drop_last=False):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.bucket_size = bucket_size
#         self.shuffle = shuffle
#         self.drop_last = drop_last
#
#     def create_batches(self):
#         indices = np.argsort([len(item['vid']) for item in self.dataset])   # 返回升序排序的索引
#         buckets = [indices[i:i + self.bucket_size] for i in range(0, len(indices), self.bucket_size)]
#         if self.shuffle:
#             np.random.shuffle(buckets)
#         batches = []
#         for bucket in buckets:
#             if len(bucket) > self.batch_size:
#                 sub_batches = [bucket[i:i + self.batch_size] for i in range(0, len(bucket), self.batch_size)]
#                 if self.drop_last and len(sub_batches[-1]) < self.batch_size:
#                     sub_batches = sub_batches[:-1]
#                 batches.extend(sub_batches)
#             else:
#                 if not self.drop_last:
#                     batches.append(bucket)
#         if self.shuffle:
#             np.random.shuffle(batches)
#         return batches
#
#     def __iter__(self):
#         batches = self.create_batches()
#         for batch in batches:
#             yield batch
#
#     def __len__(self):
#         return len(self.create_batches())


'''
class GRIDDataset(Dataset):
    def __init__(self, data, phase='train'):
        if isinstance(data, str):
            self.dataset = self.get_data_file(data)
        else:
            self.dataset = data
        print(len(self.dataset))
        self.phase = phase
        self.vocab = [PAD] + [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                                  'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']    # 28
        self.max_vid_len = 75
        self.max_txt_len = 50

    # 得到所有speaker文件目录的list (一个包含多个不同speaker的文件夹)
    def get_data_file(self, root_path):
        # GRID\LIP_160x80\lip\s1
        dataset = []
        unseen_spk = ['s1', 's2', 's20', 's21', 's22']
        for spk in os.listdir(root_path):  # 根目录下的speaker目录
            if spk in unseen_spk:
                continue
            spk_path = os.path.join(root_path, spk)
            for fn in os.listdir(spk_path):  # 1000
                data_path = os.path.join(spk_path, fn)
                if len(os.listdir(data_path)) == 75:
                    dataset.append(data_path)
        return dataset

    def load_video(self, fn):
        files = os.listdir(fn)
        files = list(filter(lambda f: f.endswith('.jpg'), files))
        files = sorted(files, key=lambda f: int(os.path.splitext(f)[0]))
        array = [cv2.imread(os.path.join(fn, f), 0) for f in files]  # 单通道
        # array = list(filter(lambda im: im is not None, array))
        array = [cv2.resize(img, (128, 64)) for img in array]
        array = np.stack(array, axis=0)[:, None].astype(np.float32)  # TCHW  C=1
        return array / 255.

    def load_txt(self, fn):
        with open(fn, 'r', encoding='utf-8') as f:
            txt = [line.strip().split(' ')[2] for line in f]
            txt = list(filter(lambda s: not s.upper() in ['SIL', 'SP'], txt))
        raw_txt = ' '.join(txt).upper()
        return np.asarray([self.vocab.index(c) for c in raw_txt])

    def padding(self, array, max_len):
        if len(array) >= max_len:
            return array[:max_len]
        return np.concatenate([array, np.zeros([max_len - len(array)] + list(array[0].shape), dtype=array[0].dtype)])

    def __getitem__(self, idx):
        item = self.dataset[idx]
        vid = self.load_video(item)
        if self.phase == 'train':
            vid = HorizontalFlip(vid, 0.5)
        txt_path = item.replace('lip', 'align_txt') + '.align'
        txt = self.load_txt(txt_path)
        vid_len = min(len(vid), self.max_vid_len)
        txt_len = min(len(txt), self.max_txt_len)
        vid = self.padding(vid, self.max_vid_len)
        txt = self.padding(txt, self.max_txt_len)
        return dict(vid=torch.FloatTensor(vid),  # (T, C, H, W)
                    txt=torch.LongTensor(txt),
                    vid_lens=torch.tensor(vid_len),
                    txt_lens=torch.tensor(txt_len))

    def __len__(self):
        return len(self.dataset)
'''


class Speaker(object):
    def __init__(self, data):
        # GRID\LIP_160x80\lip\s1\bbaf4p
        self.data = data
        self.vocab = [PAD] + [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']   # 28
        self.max_vid_len = 75
        self.max_txt_len = 50

    def sample_batch_data(self, bs):
        vids = []
        txts = []
        vid_lens = []
        txt_lens = []
        batch_paths = np.random.choice(self.data, size=bs, replace=False)  # 不重复采样
        for path in batch_paths:
            vid = self.load_video(path)
            txt_path = path.replace('lip', 'align_txt') + '.align'
            txt = self.load_txt(txt_path)
            vid_lens.append(min(len(vid), self.max_vid_len))
            txt_lens.append(min(len(txt), self.max_txt_len))
            vids.append(self.padding(vid, self.max_vid_len))
            txts.append(self.padding(txt, self.max_txt_len))
        vids = np.stack(vids, axis=0)  # (B, T, C, H, W)
        txts = np.stack(txts, axis=0)
        return dict(vid=torch.FloatTensor(vids),  # (B, T, C, H, w)
                    txt=torch.LongTensor(txts),  # (B, L)
                    vid_lens=torch.tensor(vid_lens),  # (B, )
                    txt_lens=torch.tensor(txt_lens))  # (B, )

    def load_video(self, fn):
        files = os.listdir(fn)
        files = list(filter(lambda f: f.endswith('.jpg'), files))
        files = sorted(files, key=lambda f: int(os.path.splitext(f)[0]))
        array = [cv2.imread(os.path.join(fn, f), 0) for f in files]  # 单通道
        # array = list(filter(lambda im: im is not None, array))
        array = [cv2.resize(img, (128, 64)) for img in array]
        array = np.stack(array, axis=0)[:, None].astype(np.float32)  # TCHW  C=1
        return array / 255.

    def load_txt(self, fn):
        with open(fn, 'r', encoding='utf-8') as f:
            txt = [line.strip().split(' ')[2] for line in f]
            txt = list(filter(lambda s: not s.upper() in ['SIL', 'SP'], txt))
        raw_txt = ' '.join(txt).upper()
        return np.asarray([self.vocab.index(c) for c in raw_txt])

    def padding(self, array, max_len):
        if len(array) >= max_len:
            return array[:max_len]
        return np.concatenate([array, np.zeros([max_len - len(array)] + list(array[0].shape), dtype=array[0].dtype)])



class NoiseDataset(object):
    def __init__(self, noise_path=None, sr=16000):
        if noise_path is None:  # default gaussian 
            self.noise = np.random.randn(sr*60)   # 60s
            #self.noise = np.random.normal(0, 1, sr*60)
        elif noise_path.endswith(('.txt', '.csv')):
            with open(noise_path, 'r', encoding='utf-8') as fr:
                paths = [f.strip() for f in fr if f.strip() != '']
                speeches = np.random.choice(paths, min(len(paths), 10000), replace=False)
                self.noise = [librosa.load(f, sr=sr)[0] for f in speeches]
                print('Loading speech noise ..')
        else:
            self.noise, sr = librosa.load(noise_path, sr=sr)
            print('Noise Data:', self.noise.shape, sr)
        
        #self.snrs = list(range(-10, 25, 5))   # -10 to 20
        #self.snrs = list(range(-5, 25, 5))   # -5 to 20
        #self.snrs = np.arange(-12.5, 20, 5.).tolist()   # -12.5 to 17.5  best choice
        self.snrs = np.arange(-7.5, 20, 5.).tolist()   # -7.5 to 17.5  best choice
        print('SNR Range:', self.snrs)

    def testing_noisy_signal(self, signal, snr_db=None):
        if snr_db is None: return normalize(signal)

        if len(self.noise) < len(signal):
            noise = np.tile(self.noise, len(signal) // len(self.noise) + 1)[:len(signal)]
        else:
            noise = self.noise[:len(signal)]

        SNR = 10. ** (snr_db / 10.)
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        target_noise = noise * np.sqrt(signal_power / (SNR * noise_power))
        corrupted_signal = signal + target_noise
        return normalize(corrupted_signal)

    def training_noisy_signal(self, signal, snr_range=None, p=0.25):
        if p <= 0:
            snr_db = random.choice((self.snrs if snr_range is None else snr_range) + [None])
            if snr_db is None: return normalize(signal)
        else:
            if np.random.rand() >= p: return normalize(signal)
            snr_db = random.choice(self.snrs if snr_range is None else snr_range)  
            #snr_db = round(np.random.normal(0, 5))   # -15 to 15

        if isinstance(self.noise, (list, tuple)):
            noise = random.choice(self.noise)
        else:
            noise = self.noise

        if len(noise) < len(signal):
            noise = np.tile(noise, len(signal) // len(noise) + 1)
        #loc = np.random.randint(0, len(noise) - len(signal) + 1)
        loc = np.random.randint(len(signal), len(noise) - len(signal) + 1)
        noise = noise[loc: loc + len(signal)]

        SNR = 10. ** (snr_db / 10.)
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        target_noise = noise * np.sqrt(signal_power / (SNR * noise_power))
        
        #if np.random.rand() < 0.2:  # part noise
        #    seg_len = int(np.random.uniform(0, 0.5) * len(target_noise))
        #    s0 = np.random.randint(0, len(target_noise) - seg_len)
        #    target_noise[s0: s0 + seg_len] = 0.
        
        corrupted_signal = signal + target_noise
        return normalize(corrupted_signal)



def normalize(x, norm='peak_norm'):
    if norm == 'z_score':
        mean, std = np.mean(x), np.std(x)
        if std == 0: std = 1.
        return (x - mean) / std
    elif norm == 'peak_norm':
        peak = max(1., np.max(np.abs(x)))
        return x / peak
    elif norm == 'rms_norm':
        rms = np.sqrt(np.mean(x**2))
        return x / max(rms, 1e-8)   # target_rms = 1.
    elif norm == 'max_min':
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    elif norm == 'log_norm':
        return np.log1p(x)
    elif norm is None:
        return x
    else:
        raise ValueError('Unknown Normalization!!')


'''
def vid_random_erasing(imgs, scale=(0.1, 0.33)):  # TCHW
    vid = imgs.copy()
    T, C, H, W = vid.shape
    s = random.uniform(*scale)
    h = int(s * min(H, W))
    w = int(s * min(H, W))
    x = random.randint(0, H - h)
    y = random.randint(0, W - w)
    t_s = random.randint(0, T-1)
    t_e = random.randint(t_s, T)
    # 所有时间步和通道上遮挡的值相同
    # vid[t_s:t_e, :, x:x+h, y:y+w] = np.random.uniform(0, 1)  # 0
    # 所有时间步上遮挡的值相同
    vid[t_s:t_e, :, x:x+h, y:y+w] = np.random.rand(C, h, w)  # 每一帧遮挡相同
    return vid
'''


#def vid_random_erasing(vid, p=0.5, scale=(0.1, 0.33), ratio=(0.3, 1), value=0):  # imgs: (T, C, H, W)
def vid_random_erasing(vid, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):  # imgs: (T, C, H, W)
    """
    视频随机擦除增强 (时空一致性版本)
    Args:
        vid: 输入视频张量 (T, C, H, W)
        p: 执行概率
        scale: 擦除面积比例范围
        ratio: 擦除区域宽高比范围
        value: 填充值 (int/float/tuple)
    Returns:
        增强后的视频张量
    """
    if np.random.uniform() >= p:
        return vid
    
    imgs = vid.copy()
    T, C, H, W = imgs.shape
    img_area = H * W
    # 随机生成擦除面积
    erase_area = img_area * np.random.uniform(scale[0], scale[1])
    # aspect_ratio = np.random.uniform(*ratio)
    aspect_ratio = np.exp(np.random.uniform(np.log(ratio[0]), np.log(ratio[1])))
    # 计算擦除区域尺寸
    h = int(round(np.sqrt(erase_area * aspect_ratio)))
    w = int(round(np.sqrt(erase_area / aspect_ratio)))
    # 尺寸边界检查
    h, w = min(h, H-1), min(w, W-1)
    if h == 0 or w == 0:
        return imgs
    
    # 时间轴随机选择策略
    # if np.random.rand() < 0.7:  # 70%概率全时段擦除
    #     t_start, t_end = 0, T
    # else:  # 30%概率部分时段擦除
    #     t_duration = max(1, int(T * np.random.uniform(0.1, 0.5)))
    #     t_start = np.random.randint(0, T - t_duration)
    #     t_end = t_start + t_duration
    t_dur = int(T * np.random.uniform(0, 0.4))
    t_s = np.random.randint(0, T - t_dur)
    t_e = t_s + t_dur
    
    top, left = (H - h)//2, (W - w)//2   # 中心位置
    # top, left = np.random.randint(0, H - h), np.random.randint(0, W - w)  # 随机生成位置
    # 时空一致性擦除（所有帧相同位置）
    imgs[t_s: t_e, :, top:top+h, left:left+w] = value  # 固定值
    #imgs[..., top:top+h, left:left+w] = np.random.rand(C, h, w)  # 随机值
    #imgs[t_s: t_e, :, top:top+h, left:left+w] = np.mean(imgs[t_s: t_e, :, top:top+h, left:left+w], axis=(1, 2, 3), keepdims=True)  # 均值
    #for c in range(C): imgs[:, c, top:top+h, left:left+w] = np.random.rand()  # 单个随机值 [0, 1)
    return imgs



# 高斯噪声、高斯模糊、随机遮挡(random erasing)、[time masking]
def vid_seq_noise(imgs, freq=1, p=0.5):  # 经过/255归一化的灰度图像序列 (T, C, H, W)
    if np.random.uniform() > p:
        return imgs
    
    img_seq = imgs.copy()
    if freq == 1:
        len = img_seq.shape[0]
        occ_len = random.randint(int(len * 0.1), int(len * 0.5))
        start_fr = random.randint(0, len-occ_len)
        raw_sequence = img_seq[start_fr: start_fr + occ_len]
        prob = random.random()
        if prob < 0.5:
            var = random.random() * 0.2
            raw_sequence = random_noise(raw_sequence, mode='gaussian', mean=0, var=var, clip=True)   # 添加高斯噪声，其他：椒盐噪声
        elif prob < 1.0:
            blur = torchvision.transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0))  # (..., C, H, W)
            raw_sequence = blur(torch.tensor(raw_sequence)).numpy()
        else:
            #raw_sequence = 0.
            pass
        img_seq[start_fr: start_fr + occ_len] = raw_sequence
    else:
        len_global = img_seq.shape[0]
        len = img_seq.shape[0] // freq
        for j in range(freq):
            try:
                occ_len = random.randint(int(len_global * 0.3), int(len_global * 0.5))
                start_fr = random.randint(0, len*j + len - occ_len)
                if start_fr < len*j:
                    assert 1==2
            except:
                occ_len = len // 2
                start_fr = len * j
            raw_sequence = img_seq[start_fr: start_fr + occ_len]
            prob = random.random()
            if prob < 0.5:
                var = random.random() * 0.2
                raw_sequence = random_noise(raw_sequence, mode='gaussian', mean=0, var=var, clip=True)
            elif prob < 1.0:
                blur = torchvision.transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0))   # (..., C, H, W)
                raw_sequence = blur(torch.tensor(raw_sequence)).numpy()
            else:
                #raw_sequence = 0.
                pass
            img_seq[start_fr: start_fr + occ_len] = raw_sequence
    return img_seq



class GRIDDataset(Dataset):
    max_vid_len = 75
    max_aud_len = 4 * max_vid_len
    
    def __init__(self, root_path, data_path, sample_size=2, phase='train', setting='unseen'):
        self.sample_size = sample_size  # 每个speaker采的样本数
        self.root_path = root_path
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.vocab = [PAD] + list(' '+string.ascii_uppercase) + [EOS, BOS]   # 30
        #with open('word_vocab.txt', 'r', encoding='utf-8') as fin:
        #    vocab = [line.strip() for line in fin if line.strip() != '']
        #self.vocab = [PAD] + vocab + [EOS, BOS]  # 54
        print('data path:', root_path, data_path)

        with open(data_path, 'r', encoding='utf-8') as fr:
            self.spk_dict = json.load(fr)
            # self.spks = list(self.spk_dict.keys())

        # totally 34 and #21 is missing
        self.spks = [f's{i}' for i in range(1, 35) if (setting == 'seen' and i != 21) or (setting == 'unseen' and i not in [1, 2, 20, 21, 22])]

        if self.phase == 'drl_train':
            self.data = self.spks
        else:
            self.data = []
            for spk_id in self.spk_dict.keys():
                #self.data.extend([os.path.join(self.root_path, spk_id, sd) for sd in self.spk_dict[spk_id]])
                self.data.extend([(self.root_path, spk_id, sd) for sd in self.spk_dict[spk_id]])
        print(len(self.data), len(self.spks))

        self.noise_generator = {
                                #'gauss': NoiseDataset(),
                                'white': NoiseDataset(noise_path='../NoiseDataset/NoiseX-92/white.wav'),
                                'pink': NoiseDataset(noise_path='../NoiseDataset/NoiseX-92/pink.wav'),
                                'factory1': NoiseDataset(noise_path='../NoiseDataset/NoiseX-92/factory1.wav'),
                                'factory2': NoiseDataset(noise_path='../NoiseDataset/NoiseX-92/factory2.wav'),
                                'babble': NoiseDataset(noise_path='../NoiseDataset/NoiseX-92/babble.wav'),
                                #'speech': NoiseDataset(noise_path='sidespeaker.txt'),   # speech noise
                               }
        
        self.noise_ratio = 0.25
        self.snr_range = None

    def step_noise_ratio(self, max_epoch=100):
        self.noise_ratio = min(0.25, self.noise_ratio + 0.01)
        print('Current Noise Ratio:', self.noise_ratio)

    def step_snr_range(self, ep, max_ep=50):
        if ep < int(max_ep * 0.1):
            self.snr_range = [None]
            self.noise_ratio = 0.
        elif ep < int(max_ep * 0.3):
            self.snr_range = np.arange(12.5, 20, 5.).tolist()   # 12.5 to 17.5  
            self.noise_ratio = 0.25
        elif ep < int(max_ep * 0.5):
            self.snr_range = np.arange(2.5, 20, 5.).tolist()   # 2.5 to 17.5  
        else:
            self.snr_range = np.arange(-7.5, 20, 5.).tolist()   # -7.5 to 17.5
        print('SNRs:', self.snr_range)

    def load_video(self, fn):
        '''
        files = os.listdir(fn)
        files = list(filter(lambda f: f.endswith('.jpg'), files))
        files = sorted(files, key=lambda f: int(os.path.splitext(f)[0]))
        array = [cv2.imread(os.path.join(fn, f), 0) for f in files]  # 单通道
        # array = [cv2.imread(os.path.sep.join([fn, f]), 0) for f in files]  # 单通道
        # array = list(filter(lambda im: im is not None, array))
        array = [cv2.resize(img, (128, 64)) for img in array]  # W, H
        array = np.stack(array, axis=0)[:, None].astype(np.float32)  # TCHW  C=1
        '''
        vid = np.load(fn+'.npy')  # THW
        clean_vid = vid[:, None].astype(np.float32) / 255.  # TCHW C=1
        clean_vid = self.vid_data_augment(clean_vid)  # 数据增强
        if self.phase == 'train':
            ## additive noise + blur
            #freq = random.choice([1, 2, 3])
            #noisy_vid = vid_seq_noise(clean_vid, freq, p=self.noise_ratio)
            
            ## random erasing
            #noisy_vid = vid_random_erasing(clean_vid, p=self.noise_ratio)
            
            noisy_vid = vid_time_masking(clean_vid, ratio=0.1, p=self.noise_ratio)
            #noisy_vid = vid_dropout(clean_vid, ratio=0.2, p=self.noise_ratio)
            #noisy_vid = clean_vid  # bad
        else:
            noisy_vid = clean_vid
        return clean_vid, noisy_vid

    def load_audio(self, fn, sr=16000):
        y, sr = librosa.load(fn, sr=sr)  # 16kHz
        #y, sr0 = librosa.load(fn, sr=None)
        #if sr0 != sr:  y = librosa.resample(y, orig_sr=sr0, target_sr=sr)

        #if self.phase == 'train':
        #    y = self.noise_generator['pink'].training_noisy_signal(y, p=self.noise_ratio)
        #else:
        #    y = normalize(y)   # testing for clean
        #    # y = self.noise_generator['babble'].testing_noisy_signal(y, 5)   # testing for 5dB
        #return self.get_fbank(y, sr)
        clean_aud = self.get_fbank(normalize(y), sr)
        if self.phase == 'train':
            noise_type = random.choice(list(self.noise_generator.keys()))
            noise_gen = self.noise_generator[noise_type]
            #noisy_aud = self.get_fbank(noise_gen.training_noisy_signal(y, p=self.noise_ratio), sr)
            noisy_aud = self.get_fbank(noise_gen.training_noisy_signal(y, self.snr_range, p=self.noise_ratio), sr)
        else:
            #noisy_aud = clean_aud
            noisy_aud = self.get_fbank(self.noise_generator['babble'].testing_noisy_signal(y, -5), sr) 
        return clean_aud, noisy_aud

    def get_fbank(self, y, sr=16000, norm=True):
        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, win_length=400, hop_length=160, n_mels=80)    # (25ms/10ms)
        #melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, win_length=640, hop_length=160, n_mels=80)  # (40ms/10ms)
        #melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
        log_mel = librosa.power_to_db(melspec, ref=np.max)   # (F, T) 转换到对数刻度
        if norm:
            #log_mel = (log_mel - log_mel.mean()) / (log_mel.std()+1e-8)  # z-norm
            log_mel = (log_mel - log_mel.mean(1, keepdims=True)) / (log_mel.std(1, keepdims=True)+1e-8)  # z-norm
        return log_mel.T  # (T, n_mels)

    def load_txt(self, fn):
        with open(fn, 'r', encoding='utf-8') as f:
            txt = [line.strip().split(' ')[2].upper() for line in f]
            txt = list(filter(lambda s: not s in ['SIL', 'SP'], txt))
        #return np.asarray([self.vocab.index(c) for c in ' '.join(txt).upper()])
        #return np.asarray([self.vocab.index(w.upper()) for w in txt])
        #return np.asarray([self.vocab.index(BOS)] + [self.vocab.index(w.upper()) for w in txt] + [self.vocab.index(EOS)])
        return np.asarray([self.vocab.index(c) for c in [BOS]+list(' '.join(txt))+[EOS]])

    def vid_data_augment(self, vid):
        if self.phase == 'train' and np.random.rand() < 0.5:
            return horizontal_flip(vid)
        return vid
        
    def aud_data_augment(self, aud):
        if self.phase == 'train' and np.random.rand() < 0.5:
            return spec_augment(aud, time_first=True)
        return aud

    def padding(self, array, max_len):
        if len(array) >= max_len:
            return array[:max_len]
        return np.concatenate([array, np.zeros([max_len - len(array)] + list(array[0].shape), dtype=array[0].dtype)])
        #return np.concatenate([array, np.zeros((max_len-len(array),) + array.shape[1:], dtype=array[0].dtype)])
    
    def fetch_data(self, vid_path, aud_path, align_path):
        clean_vid, noisy_vid = self.load_video(vid_path)
        clean_aud, noisy_aud = self.load_audio(aud_path)
        txt = self.load_txt(align_path)
        #vid = self.vid_data_augment(vid)
        return clean_vid, noisy_vid, clean_aud, noisy_aud, txt
    
    def get_one_data(self, idx):
        #vid_path = self.data[idx]
        #spk = vid_path.split(os.path.sep)[-2]
        root_path, spk, dir_path = self.data[idx]
        spk_id = self.spks.index(spk) if spk in self.spks else -1
        vid_path = os.path.join(root_path, spk, dir_path)
        txt_path = os.path.join(root_path.replace('lip', 'align_txt'), spk, dir_path+'.align')
        aud_path = os.path.join(root_path.replace('lip', 'audio'), spk, dir_path+'.wav')
        #txt_path = vid_path.replace('lip', 'align_txt') + '.align'
        #aud_path = vid_path.replace('lip', 'audio') + '.wav'
        clean_vid, noisy_vid, clean_aud, noisy_aud, txt = self.fetch_data(vid_path, aud_path, txt_path)
        return dict(
                    clean_vid=torch.FloatTensor(clean_vid),  # (T, C, H, W)
                    noisy_vid=torch.FloatTensor(noisy_vid),  
                    clean_aud=torch.FloatTensor(clean_aud),  # (T, C)
                    noisy_aud=torch.FloatTensor(noisy_aud),
                    txt=torch.LongTensor(txt),
                    spk_id=spk_id
                )

    '''
    def fetch_data(self, vid_path, aud_path, align_path):
        clean_vid, noisy_vid = self.load_video(vid_path)
        raw_wav, clean_aud, noisy_aud = self.load_audio(aud_path)
        txt = self.load_txt(align_path)
        #vid = self.vid_data_augment(vid)
        vid_len = min(len(clean_vid), self.max_vid_len)
        aud_len = min(len(clean_aud), self.max_aud_len)
        txt_len = min(len(txt), self.max_txt_len) - 1  # excluding bos 
        clean_vid = self.padding(clean_vid, self.max_vid_len)
        noisy_vid = self.padding(noisy_vid, self.max_vid_len)
        clean_aud = self.padding(clean_aud, self.max_aud_len)
        noisy_aud = self.padding(noisy_aud, self.max_aud_len)
        txt = self.padding(txt, self.max_txt_len)
        return raw_wav, clean_vid, noisy_vid, clean_aud, noisy_aud, txt, vid_len, aud_len, txt_len

    def get_one_data(self, idx):
        #vid_path = self.data[idx]
        #spk = vid_path.split(os.path.sep)[-2]
        root_path, spk, dir_path = self.data[idx]
        spk_id = self.spks.index(spk) if spk in self.spks else -1
        vid_path = os.path.join(root_path, spk, dir_path)
        txt_path = os.path.join(root_path.replace('lip', 'align_txt'), spk, dir_path+'.align')
        aud_path = os.path.join(root_path.replace('lip', 'audio'), spk, dir_path+'.wav')
        #txt_path = vid_path.replace('lip', 'align_txt') + '.align'
        #aud_path = vid_path.replace('lip', 'audio') + '.wav'
        raw_wav, clean_vid, noisy_vid, clean_aud, noisy_aud, txt, vid_len, aud_len, txt_len = self.fetch_data(vid_path, aud_path, txt_path)
        return dict(
                    raw_wav=raw_wav,
                    clean_vid=torch.FloatTensor(clean_vid),  # (T, C, H, W)
                    noisy_vid=torch.FloatTensor(noisy_vid),  
                    clean_aud=torch.FloatTensor(clean_aud),  # (T, C)
                    noisy_aud=torch.FloatTensor(noisy_aud),
                    txt=torch.LongTensor(txt),
                    spk_id=spk_id,
                    vid_lens=vid_len,
                    aud_lens=aud_len,
                    txt_lens=txt_len
                 )
    '''

    # 返回一个speaker的数据
    def get_one_speaker(self, idx):  # one batch speaker data
        vids = []
        auds = []
        txts = []
        vid_lens = []
        aud_lens = []
        txt_lens = []
        # GRID\LIP_160x80\lip\s1
        spk_id = self.data[idx]
        # GRID\LIP_160x80\lip\s1\bbaf4p
        spk_data = [os.path.join(self.root_path, spk_id, sd) for sd in self.spk_dict[spk_id]]
        batch_data = np.random.choice(spk_data, size=self.sample_size, replace=False)  # 不重复采样
        for vid_path in batch_data:
            txt_path = vid_path.replace('lip', 'align_txt') + '.align'
            aud_path = vid_path.replace('lip', 'audio') + '.wav'
            vid, aud, txt, vid_len, aud_len, txt_len = self.fetch_data(vid_path, aud_path, txt_path)
            vids.append(vid)
            auds.append(aud)
            txts.append(txt)
            vid_lens.append(vid_len)
            aud_lens.append(aud_len)
            txt_lens.append(txt_len)
        vids = np.stack(vids, axis=0)  # (N, T, C, H, W)
        auds = np.stack(auds, axis=0)  
        txts = np.stack(txts, axis=0)
        return dict(vid=torch.FloatTensor(vids),  # (N, T, C, H, w)
                    aud=torch.FloatTensor(auds),
                    txt=torch.LongTensor(txts),  # (N, L)
                    vid_lens=torch.LongTensor(vid_lens),  # (N, )
                    aud_lens=torch.LongTensor(aud_lens),
                    txt_lens=torch.LongTensor(txt_lens))  # (N, )

    def __getitem__(self, idx):
        if self.phase == 'drl_train':
            return self.get_one_speaker(idx)
        else:
            return self.get_one_data(idx)

    def __len__(self):
        return len(self.data)

    @classmethod
    def collate_pad(cls, batch):
        #return torch.utils.data.dataloader.default_collate(batch)
        def padding(tensor, max_len):
            # tensor: shape (L, ...)
            L = tensor.size(0)
            if L >= max_len: return tensor[:max_len]
            pad_shape = (max_len-L,) + tensor.shape[1:]
            pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, pad], dim=0)

        clean_vids, clean_auds, noisy_vids, noisy_auds, txts, spk_ids = [], [], [], [], [], []
        vid_lens, aud_lens, txt_lens = [], [], []
        for d in batch:
            clean_vids.append(d['clean_vid'])
            clean_auds.append(d['clean_aud'])
            noisy_vids.append(d['noisy_vid'])
            noisy_auds.append(d['noisy_aud'])
            txts.append(d['txt'])
            #spk_ids.append(d['spk_id'])
            vid_lens.append(min(len(d['clean_vid']), cls.max_vid_len))
            aud_lens.append(min(len(d['clean_aud']), cls.max_aud_len))
            txt_lens.append(len(d['txt']) - 1)     # excluding bos
        pad_batch = dict()
        pad_batch['clean_vid'] = pad_sequence(clean_vids, batch_first=True)[:, :cls.max_vid_len]  # (T, C, H, W)
        pad_batch['clean_aud'] = pad_sequence(clean_auds, batch_first=True)[:, :cls.max_aud_len]  # (T, C)
        pad_batch['noisy_vid'] = pad_sequence(noisy_vids, batch_first=True)[:, :cls.max_vid_len]
        pad_batch['noisy_aud'] = pad_sequence(noisy_auds, batch_first=True)[:, :cls.max_aud_len]
        pad_batch['txt'] = pad_sequence(txts, batch_first=True)
        #pad_batch['spk_id'] = torch.tensor(spk_ids)
        pad_batch['vid_lens'] = torch.tensor(vid_lens)
        pad_batch['aud_lens'] = torch.tensor(aud_lens)
        pad_batch['txt_lens'] = torch.tensor(txt_lens)
        return pad_batch



class CMLRDataset(Dataset):
    # 类变量
    MAX_VID_LEN = 200
    MAX_AUD_LEN = 800
    MAX_TXT_LEN = 40

    def __init__(self, root_path, file_list, phase='train', setting='unseen'):
        self.root_path = root_path
        self.phase = phase
        self.spks = [f's{i}' for i in range(1, 12) if setting == 'seen' or (setting == 'unseen' and i not in [2, 6])]
        self.data = []
        with open(file_list, 'r', encoding='utf-8') as f:
            for line in f:  # s5/20151009_section_3_030.36_032.65
                spk_id, section = line.strip().split('/')
                date, sec_id = section.split('_', 1)   # 20151009  section_3_030.36_032.65
                self.data.append((spk_id, date, sec_id))
        print('data path:', root_path, file_list)
        print(len(self.data), len(self.spks))

        with open('zh_vocab.txt', 'r', encoding='utf-8') as fin:
            vocab = [line.strip() for line in fin if line.strip() != '']
        #self.vocab = [PAD] + vocab
        self.vocab = [PAD] + vocab + [EOS, BOS]

        #self.noise_generator = NoiseDataset(noise_path='../NoiseDataset/NoiseX-92/pink.wav')
        # self.noise_generator = NoiseDataset(noise_path='../NoiseDataset/NoiseX-92/babble.wav')
        # self.noise_generator = NoiseDataset(noise_path='../NoiseDataset/NoiseX-92/white.wav')
        self.noise_generator = {
                                #'gauss': NoiseDataset(),
                                'white': NoiseDataset(noise_path='../NoiseDataset/NoiseX-92/white.wav'),
                                'pink': NoiseDataset(noise_path='../NoiseDataset/NoiseX-92/pink.wav'),
                                'factory1': NoiseDataset(noise_path='../NoiseDataset/NoiseX-92/factory1.wav'),
                                'factory2': NoiseDataset(noise_path='../NoiseDataset/NoiseX-92/factory2.wav'),
                                'babble': NoiseDataset(noise_path='../NoiseDataset/NoiseX-92/babble.wav')
                               }
        
        self.noise_ratio = 0.3


    def load_video(self, fn):
        '''
        files = os.listdir(fn)
        files = list(filter(lambda f: f.endswith('.jpg'), files))
        files = sorted(files, key=lambda f: int(os.path.splitext(f)[0]))
        array = [cv2.imread(os.path.join(fn, f), 0) for f in files]  # 单通道
        # array = list(filter(lambda im: im is not None, array))
        # array = [cv2.resize(img, (128, 64)) for img in array if img.shape[:2] != (64, 128) else img]  # W, H
        array = np.stack(array, axis=0)[:, None].astype(np.float32)  # TCHW  C=1
        return array / 255.
        '''
        vid = np.load(fn+'.npy')
        #return vid[:, None].astype(np.float32) / 255.  # TCHW  C=1
        clean_vid = vid[:, None].astype(np.float32) / 255.  # TCHW C=1
        clean_vid = self.vid_data_augment(clean_vid)  # 数据增强
        if self.phase == 'train':
            ## additive noise + blur
            #freq = random.choice([1, 2, 3])
            #noisy_vid = vid_seq_noise(clean_vid, freq, p=self.noise_ratio)
            
            ## random erasing
            #noisy_vid = vid_random_erasing(clean_vid, p=self.noise_ratio)
            
            noisy_vid = vid_time_masking(clean_vid, ratio=0.1, p=self.noise_ratio)
            #noisy_vid = vid_dropout(clean_vid, ratio=0.2, p=self.noise_ratio)
            #noisy_vid = clean_vid  # bad
        else:
            noisy_vid = clean_vid
        return clean_vid, noisy_vid

    '''
    def load_audio(self, fn, sr=16000):
        # y, sr = librosa.load(fn, sr=sr)  # 16kHz
        y, sr0 = librosa.load(fn, sr=None)
        if sr0 != sr:
            y = librosa.resample(y, orig_sr=sr0, target_sr=sr)
        if self.phase == 'train':
            y = self.noise_generator.training_noisy_signal(y, p=0.25)
        else:
            y = normalize(y)
            # y = self.noise_generator.testing_noisy_signal(y, 5)  # 5dB
        # melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, win_length=400, hop_length=160, n_mels=80)  # 25ms win / 10ms hop for grid
        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=80)   # 128ms win / 32ms hop for cmlr
        # melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)   # 64ms win / 16ms hop for cmlr
        #melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, win_length=800, hop_length=160, n_mels=80)
        logmelspec = librosa.power_to_db(melspec, ref=np.max)
        #return logmelspec.T  # (T, n_mels)
        log_mel = logmelspec.T   # (T, n_mels)
        norm_log_mel = np.nan_to_num((log_mel - log_mel.mean(0, keepdims=True)) / (log_mel.std(0, keepdims=True)+1e-8))  # z-norm
        return norm_log_mel
    '''
     
    def load_audio(self, fn, sr=16000):
        y, sr = librosa.load(fn, sr=sr)  # 16kHz
        #y, sr0 = librosa.load(fn, sr=None)
        #if sr0 != sr: y = librosa.resample(y, orig_sr=sr0, target_sr=sr)
        
        #if self.phase == 'train':
        #    y = self.noise_generator['pink'].training_noisy_signal(y, p=self.noise_ratio)
        #else:
        #    y = normalize(y)   # testing for clean
        #    # y = self.noise_generator['babble'].testing_noisy_signal(y, 5)   # testing for 5dB
        #return self.get_fbank(y, sr)

        clean_aud = self.get_fbank(normalize(y), sr)
        if self.phase == 'train':
            cls = random.choice(list(self.noise_generator.keys()))
            noisy_aud = self.get_fbank(self.noise_generator[cls].training_noisy_signal(y, p=self.noise_ratio), sr)
        else:
            noisy_aud = clean_aud
            #noisy_aud = self.get_fbank(self.noise_generator['babble'].testing_noisy_signal(y, 15), sr)  
        return clean_aud, noisy_aud

    def get_fbank(self, y, sr=16000, norm=True):
        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, win_length=400, hop_length=160, n_mels=80)    # (25ms/10ms)
        #melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, win_length=640, hop_length=160, n_mels=80)  # (40ms/10ms)
        #melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=320, n_mels=80)   # 32ms / 20ms
        #melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)   # 64ms win / 16ms hop for cmlr
        log_mel = librosa.power_to_db(melspec, ref=np.max)   # (F, T) 转换到对数刻度
        if norm:
            #log_mel = (log_mel - log_mel.mean()) / (log_mel.std()+1e-8)  # z-norm
            log_mel = (log_mel - log_mel.mean(1, keepdims=True)) / (log_mel.std(1, keepdims=True)+1e-8)  # z-norm
        return log_mel.T  # (T, n_mels)

    def load_txt(self, fn):
        with open(fn, 'r', encoding='utf-8') as f:
            txt = f.readline().strip()   # 读第一行
        #return np.asarray([self.vocab.index(w) for w in txt])
        # return np.asarray([self.vocab.index(BOS)] + [self.vocab.index(w) for w in txt] + [self.vocab.index(EOS)])
        return np.asarray(list(map(self.vocab.index, [BOS]+list(txt)+[EOS])))

    def vid_data_augment(self, vid):
        if self.phase == 'train' and np.random.rand() < 0.5:
            return horizontal_flip(vid)
        return vid
        
    def aud_data_augment(self, aud):
        if self.phase == 'train' and np.random.rand() < 0.5:
            return spec_augment(aud, time_first=True)
        return aud

    def data_augment(self, vid, aud):
        if self.phase == 'train' and np.random.rand() < 0.5:
            vid = horizontal_flip(vid)
            #aud = spec_augment(aud, time_first=True)
            return vid, aud
        return vid, aud

    def padding(self, array, max_len):
        if len(array) >= max_len:
            return array[:max_len]
        return np.concatenate([array, np.zeros([max_len - len(array)] + list(array[0].shape), dtype=array[0].dtype)])

    # def fetch_data(self, vid_path, aud_path, align_path):
    #     vid = self.load_video(vid_path)
    #     aud = self.load_audio(aud_path)
    #     txt = self.load_txt(align_path)
    #     # data augmentation
    #     vid, aud = self.data_augment(vid, aud)
    #     #print(vid.shape, aud.shape, len(txt), flush=True)
    #     vid_len = min(len(vid), self.MAX_VID_LEN)
    #     aud_len = min(len(aud), self.MAX_AUD_LEN)
    #     txt_len = min(len(txt), self.MAX_TXT_LEN) - 1  # excluding bos 
    #     vid = self.padding(vid, self.MAX_VID_LEN)
    #     aud = self.padding(aud, self.MAX_AUD_LEN)
    #     txt = self.padding(txt, self.MAX_TXT_LEN)
    #     return vid, aud, txt, vid_len, aud_len, txt_len

    #def fetch_data(self, vid_path, aud_path, align_path):
    #    vid = self.load_video(vid_path)
    #    aud = self.load_audio(aud_path)
    #    txt = self.load_txt(align_path)
    #    # data augmentation
    #    vid, aud = self.data_augment(vid, aud)
    #    # print(vid.shape, aud.shape, len(txt), flush=True)
    #    return vid, aud, txt
        
    def fetch_data(self, vid_path, aud_path, align_path):
        clean_vid, noisy_vid = self.load_video(vid_path)
        clean_aud, noisy_aud = self.load_audio(aud_path)
        txt = self.load_txt(align_path)
        #vid = self.vid_data_augment(vid)
        vid_len = min(len(clean_vid), self.MAX_VID_LEN)
        aud_len = min(len(clean_aud), self.MAX_AUD_LEN)
        txt_len = min(len(txt), self.MAX_TXT_LEN) - 1  # excluding bos 
        clean_vid = self.padding(clean_vid, self.MAX_VID_LEN)
        noisy_vid = self.padding(noisy_vid, self.MAX_VID_LEN)
        clean_aud = self.padding(clean_aud, self.MAX_AUD_LEN)
        noisy_aud = self.padding(noisy_aud, self.MAX_AUD_LEN)
        txt = self.padding(txt, self.MAX_TXT_LEN)
        return clean_vid, noisy_vid, clean_aud, noisy_aud, txt, vid_len, aud_len, txt_len

    #def get_one_data(self, idx):
    #    item = self.data[idx]
    #    data_path = os.path.join(*item)
    #    spk_id = self.spks.index(item[0]) if item[0] in self.spks else -1
    #    vid_path = os.path.join(self.root_path, 'video', data_path)
    #    txt_path = os.path.join(self.root_path, 'text', data_path+'.txt')
    #    aud_path = os.path.join(self.root_path, 'audio', data_path+'.wav')
    #    # vid, aud, txt, vid_len, aud_len, txt_len = self.fetch_data(vid_path, aud_path, txt_path)
    #    vid, aud, txt = self.fetch_data(vid_path, aud_path, txt_path)
    #    return dict(vid=torch.FloatTensor(vid),  # (T, C, H, W)
    #                aud=torch.FloatTensor(aud),  # (T, D)
    #                txt=torch.LongTensor(txt),   # (L, )
    #                spk_id=spk_id)
                    
    def get_one_data(self, idx):
        item = self.data[idx]
        data_path = os.path.join(*item)
        spk_id = self.spks.index(item[0]) if item[0] in self.spks else -1
        vid_path = os.path.join(self.root_path, 'video', data_path)
        txt_path = os.path.join(self.root_path, 'text', data_path+'.txt')
        aud_path = os.path.join(self.root_path, 'audio', data_path+'.wav')
        clean_vid, noisy_vid, clean_aud, noisy_aud, txt, vid_len, aud_len, txt_len = self.fetch_data(vid_path, aud_path, txt_path)
        return dict(
                    clean_vid=torch.FloatTensor(clean_vid),  # (T, C, H, W)
                    noisy_vid=torch.FloatTensor(noisy_vid),  
                    clean_aud=torch.FloatTensor(clean_aud),  # (T, C)
                    noisy_aud=torch.FloatTensor(noisy_aud),
                    txt=torch.LongTensor(txt),
                    spk_id=spk_id,
                 )

    def __getitem__(self, idx):
        return self.get_one_data(idx)

    def __len__(self):
        return len(self.data)

    # 按照batch中最长序列的长度进行padding，返回对齐后的序列和实际序列长度
    @classmethod
    def collate_pad(cls, batch):
        padded_batch = {}
        for data_type in batch[0].keys():
            if data_type == 'spk_id':
                padded_batch[data_type] = torch.tensor([s[data_type] for s in batch])
            else:
                #if data_type == 'vid':
                if 'vid' in data_type:
                    max_len = cls.MAX_VID_LEN
                #elif data_type == 'aud':
                elif 'aud' in data_type:
                    max_len = cls.MAX_AUD_LEN
                elif data_type == 'txt':
                    max_len = None
                else:
                    max_len = None
                pad_vid, ret_lens = pad_seqs3([s[data_type] for s in batch if s[data_type] is not None], max_len)
                padded_batch[data_type] = pad_vid
                if data_type == 'txt':
                    padded_batch[data_type+'_lens'] = torch.tensor(ret_lens) - 1  # excluding bos 
                else:
                    padded_batch[data_type+'_lens'] = torch.tensor(ret_lens)
        return padded_batch

    # @classmethod
    # def collate_pad(cls, batch):
    #     return torch.utils.data.dataloader.default_collate(batch)


def pad_seqs(samples, max_len=None, pad_val=0.):
    if max_len is None:
        lens = [len(s) for s in samples]
    else:
        lens = [min(len(s), max_len) for s in samples]
    max_len = max(lens)
    padded_batch = samples[0].new_full((len(samples), max_len, ) + samples[0].shape[1:], pad_val)
    for i, s in enumerate(samples):
        if len(s) < max_len:
            padded_batch[i][:len(s)] = s
        else:
            padded_batch[i] = s[:max_len]
    return padded_batch, lens


def pad_seqs2(samples, max_len=None, pad_val=0.):
    if max_len is None:
        lens = [len(s) for s in samples]
    else:
        lens = [min(len(s), max_len) for s in samples]
    max_len = max(lens)
    padded_batch = []
    for seq in samples:
        if len(seq) < max_len:
            padding = seq.new_full((max_len-len(seq), ) + seq.shape[1:], pad_val)
            padded_seq = torch.cat((seq, padding), dim=0)
        else:
            padded_seq = seq[:max_len]
        padded_batch.append(padded_seq)
    return torch.stack(padded_batch), lens


def pad_seqs3(samples, max_len=None, pad_val=0.):
    if max_len is None:
        lens = [len(s) for s in samples]
    else:
        lens = [min(len(s), max_len) for s in samples]
    max_len = max(lens)
    padded_batch = pad_sequence(samples, batch_first=True, padding_value=pad_val)  # (B, L_max, ...)
    return padded_batch[:, :min(max_len, padded_batch.shape[1])], lens
