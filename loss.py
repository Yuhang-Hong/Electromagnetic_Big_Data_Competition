import torch.nn as nn
import torch.nn.functional as F

class CosineSimilarityLoss(nn.Module):
    def __init__(self, mse_weight=0.0001):
        super(CosineSimilarityLoss, self).__init__()
        self.mse_weight = mse_weight  # MSE损失的权重系数

    def forward(self, output, target):
        """
        计算余弦相似度损失并加上 MSE 损失。
        
        参数：
          output: (batch_size, feature_dim) - 预测的向量
          target: (batch_size, feature_dim) - 真实的向量
          
        返回：
          total_loss: 组合后的总损失，包含余弦相似度损失和 MSE 损失
        """
        # 计算余弦相似度
        cosine_sim = F.cosine_similarity(output, target, dim=0)  # 计算每个样本的余弦相似度
        cosine_loss = 1 - cosine_sim.mean()  # 余弦相似度损失，越大越好

        # 计算 MSE 损失
        mse_loss_fn = nn.MSELoss()
        mse_loss = mse_loss_fn(output, target)  # 计算 MSE 损失

        # 组合损失：余弦相似度损失 + MSE 损失
        total_loss = cosine_loss + self.mse_weight * mse_loss
        
        return total_loss
