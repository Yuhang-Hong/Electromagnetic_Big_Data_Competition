import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 数据标准化（IQ信号专用）
class IQNormalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('iq_mean', torch.tensor([0.0, 0.0]))
        self.register_buffer('iq_std', torch.tensor([1.0, 1.0]))

    def forward(self, x):
        return (x - self.iq_mean.view(1,2,1)) / self.iq_std.view(1,2,1)
    
    
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, 
                     padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, 3, 
                     padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_ch)
        )
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return F.gelu(self.conv(x) + self.shortcut(x))
    

class IQSeqCompressor(nn.Module):
    def __init__(self, input_dim=2, seq_len=2016, output_dim=400):
        super().__init__()
        
        # 1. 多尺度特征提取
        # self.normalize = IQNormalize()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 64, 7, padding=3),  # [32,64,2016]
            nn.GELU(),
            nn.MaxPool1d(3, stride=2, padding=1),     # [32,64,1008]
            ResidualBlock(64, 128, dilation=2),
            nn.MaxPool1d(3, stride=2, padding=1),     # [32,128,504]
            ResidualBlock(128, 256, dilation=3),
            nn.AdaptiveAvgPool1d(400)                 # [32,256,400]
        )
        
        # 2. 时序注意力模块
        self.temporal_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=1024,
                activation='gelu',
            ),
            num_layers=4
        )
        
        # 3. 动态投影头
        self.projection = nn.Sequential(
            nn.Linear(400, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, output_dim))
        
        # 4. 输出归一化（适配余弦损失）
        self.norm = nn.LayerNorm(output_dim)
        self.relu = nn.ReLU()
    def forward(self, x, inference=False):
        # 输入形状: [32, 2, 2016]
        
        # 特征提取与压缩
        # x = self.normalize(x)
        x = self.feature_extractor(x)     # [32,256,400]
        
        # 时序建模
        x = rearrange(x, 'b c t -> t b c')  # [400,32,256]
        x = self.temporal_attn(x)           # [400,32,256]
        x = rearrange(x, 't b c -> b c t')  # [32,256,400]
        
        # 特征投影
        x = x.mean(dim=1)                   # [32,400]
        x = self.projection(x)              # [32,400]
        # x = self.norm(x)
        x = self.relu(x)
        if inference:
            return torch.round(x)
        return x                # 归一化适配余弦损失

