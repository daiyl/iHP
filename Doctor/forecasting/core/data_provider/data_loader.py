import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features

class Dataset_Custom(Dataset):
    def __init__(self,
                 args,
                 root_path,
                 flag='train',
                 size=None,
                 features='S',
                 data_path='ETTh1.csv',
                 target='OT',
                 scale=True,
                 timeenc=0,
                 freq='h',
                 seasonal_patterns=None):
        # 基于原始序列按 sample_len 步切块，每块内部滑窗 seq_len→pred_len
        self.args = args
        assert flag in ['train', 'val', 'test']
        self.flag = flag

        # 窗口长度与预测长度
        self.seq_len = size[0] if size is not None else 15
        self.pred_len = size[2] if size is not None else 1
        self.sample_len = args.sample_len  # 每个大样本步数

        self.features = features
        self.target = target
        self.scale = bool(getattr(args, "scale", scale))
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        # 读取和预处理，然后按样本块分区 train/val/test
        if self.seq_len <= 0 or self.pred_len <= 0 or self.sample_len < self.seq_len + self.pred_len:
            raise ValueError('Require positive window lengths and sample_len >= seq_len + pred_len')
        self._read_and_split_blocks()

        # 每块内部滑窗数（确保每个窗口能完整包含 seq_len + pred_len）
        if self.sample_len < (self.seq_len + self.pred_len):
            raise ValueError(
                f"sample_len ({self.sample_len}) must be >= seq_len ({self.seq_len}) + pred_len ({self.pred_len})")

        # 有效滑窗数（包含最后一个完整窗口）
        self.windows_per_sample = self.sample_len - self.seq_len - self.pred_len + 1

        # 总滑窗数
        self.total_windows = int(self.blocks.shape[0]) * int(self.windows_per_sample)

    def _read_and_split_blocks(self):
        # 1. 读取CSV
        df = pd.read_csv(os.path.join(self.root_path, self.data_path))
        cols = [c for c in df.columns if c not in ['date', self.target]]
        df = df[['date'] + cols + [self.target]]

        # 2. 提取数值矩阵
        if self.features in ['M', 'MS']:
            vals = df.iloc[:, 1:].values
        else:
            vals = df[[self.target]].values
        if not np.isfinite(vals.astype(float)).all():
            raise ValueError('CSV numerical columns contain NaN or infinity; preprocess explicitly before training')

        # 3. 构造时间特征
        stamps = df[['date']].copy()
        stamps['date'] = pd.to_datetime(stamps['date'])
        if self.timeenc == 0:
            stamps_feat = np.stack([stamps['date'].dt.month.values,
                                    stamps['date'].dt.day.values,
                                    stamps['date'].dt.weekday.values,
                                    stamps['date'].dt.hour.values], axis=1)
        else:
            # Model configs use TSLib's 't'; newer pandas no longer accepts that alias.
            offsets = {'t': pd.offsets.Minute(), 's': pd.offsets.Second(),
                       'h': pd.offsets.Hour(), 'd': pd.offsets.Day(),
                       'b': pd.offsets.BusinessDay(), 'w': pd.offsets.Week(),
                       'm': pd.offsets.MonthEnd(), 'a': pd.offsets.YearEnd()}
            tf = time_features(pd.to_datetime(stamps['date'].values), freq=offsets.get(self.freq, self.freq))
            stamps_feat = tf.transpose(1, 0)

        # 4. 标准化（训练集Block拟合）
        # 先按块切分所有数据
        if len(vals) % self.sample_len:
            raise ValueError('CSV row count must be divisible by sample_len; refusing to discard partial samples')
        n_blocks = len(vals) // self.sample_len
        if n_blocks < 10:
            raise ValueError('At least 10 independent samples are required for a nonempty 70/10/20 split')
        blocks_all = vals[:n_blocks * self.sample_len].reshape(n_blocks, self.sample_len, -1)
        stamps_all = stamps_feat[:n_blocks * self.sample_len].reshape(n_blocks, self.sample_len, -1)

        # 随机打乱块顺序然后分割
        indices = np.arange(n_blocks)
        np.random.RandomState(self.args.seed).shuffle(indices)
        n_train = int(n_blocks * 0.7)
        n_val = int(n_blocks * 0.1)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        # 根据 flag 选择对应块
        if self.flag == 'train':
            sel = train_idx
        elif self.flag == 'val':
            sel = val_idx
        else:
            sel = test_idx

        self.sample_indices = sel.copy()
        self.blocks = blocks_all[sel]
        self.stamps = stamps_all[sel]

        # 仅用训练集的完整样本拟合，各列分别进行 Z-score 标准化
        self.scaler = StandardScaler()
        if self.scale:
            train_vals = blocks_all[train_idx].reshape(-1, blocks_all.shape[-1])
            self.scaler.fit(train_vals)
            # 变换所有块
            self.blocks = self.scaler.transform(self.blocks.reshape(-1, blocks_all.shape[-1])).reshape(self.blocks.shape)

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        # 由全局窗口 idx 映射到块ID和偏移
        sample_id = idx // self.windows_per_sample
        offset    = idx % self.windows_per_sample
        block = self.blocks[sample_id]
        stamp = self.stamps[sample_id]

        x  = block[offset: offset + self.seq_len]
        y  = block[offset + self.seq_len: offset + self.seq_len + self.pred_len]
        xm = stamp[offset: offset + self.seq_len]
        ym = stamp[offset + self.seq_len: offset + self.seq_len + self.pred_len]
        return x, y, xm, ym

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data) if self.scale else data



