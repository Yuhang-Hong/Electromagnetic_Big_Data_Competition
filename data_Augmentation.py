import torch
from torch.utils.data import TensorDataset
import numpy as np
class IQAugmentation(torch.nn.Module):
    def __init__(self, noise_std=0.05, max_scale=0.1, max_phase=np.pi/8, 
                 max_freq_shift=0.005, flip_prob=0.5, iq_scale=0.1):
        super().__init__()
        self.noise_std = noise_std          # 噪声标准差
        self.max_scale = max_scale          # 最大幅度缩放比例
        self.max_phase = max_phase          # 最大相位旋转角度
        self.max_freq_shift = max_freq_shift # 最大频率偏移比例
        self.flip_prob = flip_prob           # 时间翻转概率
        self.iq_scale = iq_scale            # I/Q通道独立缩放比例

    def forward(self, x, labels=None):
        """
        x: 输入数据 [batch, 2, seq_len]
        labels: 标签 [batch, label_len]
        返回增强后的数据和标签
        """
        batch_size, _, seq_len = x.shape
        
        # 1. 添加高斯噪声
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        # 2. 全局幅度缩放
        if self.max_scale > 0:
            scale_factor = 1 + (torch.rand(batch_size, 1, 1, device=x.device) * 2 * self.max_scale - self.max_scale)
            x = x * scale_factor
        
        # 3. 相位旋转
        if self.max_phase > 0:
            theta = (torch.rand(batch_size, device=x.device) * 2 - 1) * self.max_phase  # [-max_phase, max_phase]
            cos_theta = torch.cos(theta).view(-1, 1, 1)
            sin_theta = torch.sin(theta).view(-1, 1, 1)
            # 复数旋转: I' = I*cosθ - Q*sinθ; Q' = I*sinθ + Q*cosθ
            new_i = x[:, 0, :] * cos_theta - x[:, 1, :] * sin_theta
            new_q = x[:, 0, :] * sin_theta + x[:, 1, :] * cos_theta
            x = torch.stack([new_i, new_q], dim=1)
        
        # 4. 频率偏移
        if self.max_phase > 0:
            theta = (torch.rand(batch_size, device=x.device) * 2 - 1) * self.max_phase  # [-max_phase, max_phase]
            cos_theta = torch.cos(theta).view(batch_size, 1, 1)
            sin_theta = torch.sin(theta).view(batch_size, 1, 1)
            new_i = x[:, 0, :] * cos_theta - x[:, 1, :] * sin_theta  # 结果形状为 [batch, 1, seq_len]
            new_q = x[:, 0, :] * sin_theta + x[:, 1, :] * cos_theta  # 结果形状为 [batch, 1, seq_len]
            x = torch.stack([new_i, new_q], dim=1)  # 得到形状 [batch, 2, 1, seq_len]
            x = x.squeeze(2)  # 去除中间那一维，最终形状为 [batch, 2, seq_len]
        
        # 5. 时间翻转（需同步翻转标签）
        if self.flip_prob > 0:
            flip_mask = torch.rand(batch_size, device=x.device) < self.flip_prob
            x[flip_mask] = torch.flip(x[flip_mask], dims=[-1])
            if labels is not None:
                labels[flip_mask] = torch.flip(labels[flip_mask], dims=[-1])
        
        # 6. I/Q通道独立缩放
        if self.iq_scale > 0:
            i_scale = 1 + (torch.rand(batch_size, 1, 1, device=x.device) * 2 - 1) * self.iq_scale
            q_scale = 1 + (torch.rand(batch_size, 1, 1, device=x.device) * 2 - 1) * self.iq_scale
            x = torch.stack([x[:, 0, :] * i_scale, x[:, 1, :] * q_scale], dim=1)
        
        return x, labels

# 自定义数据集类，在 __getitem__ 中进行数据增强
class AugmentedTensorDataset(TensorDataset):
    def __init__(self, *tensors, transform=None):
        super().__init__(*tensors)
        self.transform = transform

    def __getitem__(self, index):
        # 获取原始数据和标签
        x, y = super().__getitem__(index)
        # 如果定义了数据增强，则应用
        if self.transform is not None:
            # 为保证输入维度为 [batch, ...]，先增加 batch 维度
            x, y = self.transform(x.unsqueeze(0), y.unsqueeze(0))
            # 去除 batch 维度
            x = x.squeeze(0)
            y = y.squeeze(0)
        return x, y
    
