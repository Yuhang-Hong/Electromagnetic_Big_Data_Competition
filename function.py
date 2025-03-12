import torch

def calculate_cq_score_torch(y_pred, y_true):
    # 计算 CS_i
    # 逐样本计算 CS_i
    cs_i = torch.sum(y_pred * y_true, dim=0) / (torch.norm(y_pred, dim=0) * torch.norm(y_true, dim=0))

    # 按公式计算 CQ_Score
    cq_score = torch.where(cs_i < 0.7, 
                           torch.tensor(0.0, device=y_pred.device), 
                           torch.where(cs_i > 0.95, 
                                       torch.tensor(100.0, device=y_pred.device), 
                                       ((cs_i - 0.7) / 0.25) * 100))
    
    return cq_score.sum() # 计算 batch 的总和 CQ_Score

def truncate_pair(target, output):
    """
    对单个样本的 target 向量进行截断，并对对应的 output 向量做相同切断。
    
    参数：
      target: 1D torch.Tensor，形状为 [T]
      output: 1D torch.Tensor，形状为 [T]
    
    返回：
      truncated_target, truncated_output：切断后的 target 与 output
      如果 target 全为0，则返回空张量。
    """
    non_zero_indices = (target != 0).nonzero(as_tuple=False)
    if non_zero_indices.numel() == 0:
        # 如果整个 target 全为 0，则返回空张量
        return target.new_tensor([]), output.new_tensor([])
    # 找到最后一个非0元素的索引
    last_idx = non_zero_indices[-1].item()
    # 切断 target 和 output 到 last_idx+1（包含最后一个非零值）
    return target[:last_idx+1], output[:last_idx+1]

def truncate_batch_pairs(targets, outputs):
    """
    对一批样本（batch）中的 target 和 output 向量分别做截断。
    
    参数：
      targets: 2D torch.Tensor，形状为 [B, T]
      outputs: 2D torch.Tensor，形状为 [B, T]
    
    返回：
      truncated_targets, truncated_outputs：列表，每个元素是截断后的1D tensor，
      列表长度为 batch size。
    """
    truncated_targets = []
    truncated_outputs = []
    for i in range(targets.size(0)):
        t_trunc, o_trunc = truncate_pair(targets[i], outputs[i])
        truncated_targets.append(t_trunc)
        truncated_outputs.append(o_trunc)
    return truncated_targets, truncated_outputs