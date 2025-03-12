import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loss import CosineSimilarityLoss
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm 

from model.IQSeqCompressor import IQSeqCompressor
from function import * 

########################################
# 1. 准备数据：与训练时保持一致
########################################
with open('data.pkl', 'rb') as f:
    data_dict = pickle.load(f)

# 提取特征、标签和 mod_type（辅助标签）
X = data_dict['iq_data']       # IQ 信号数据，形状 [N, 2, 2016]
y = data_dict['mod_codes']     # 回归目标，形状 [N, 400]
# mod_type = data_dict['mod_types']  # 辅助标签，每个样本的调制类型，形状 [N]

# 转换为 PyTorch 张量
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)  # 如果目标为整数，这里使用 long 类型
# mod_type_tensor = torch.tensor(mod_type, dtype=torch.long)

# 划分训练集和测试集（同时拆分 X、y 和 mod_type）
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42
)

# 构建 TensorDataset（包含 3 个输入）
train_dataset = TensorDataset(X_train, y_train)
test_dataset  = TensorDataset(X_test, y_test)
batch_size=32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

########################################
# 2. 加载模型并恢复权重
########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = IQSeqCompressor().cuda()

# 加载最优权重，比如 best_model_epoch10.pth
model.load_state_dict(torch.load("best_model_epoch10.pth", map_location=device))

model.eval()

########################################
# 3. 定义损失函数（与训练相同）
########################################
criterion = CosineSimilarityLoss()

########################################
# 4. 在验证集/测试集上计算损失与 CQ_Score
########################################
val_loss = 0.0
cq_score = 0.0
count=0
    
num_batches = len(val_loader)

with torch.no_grad():
    for inputs, targets in tqdm(val_loader):
        inputs   = inputs.to(device)
        targets  = targets.float().to(device)

        # inference=True -> 输出为整数
        outputs = model(inputs, inference=False)

        # # 计算验证损失（余弦相似度损失）
        # loss = criterion(outputs, targets)
        
        # val_loss += loss.item()

        # # 计算 CQ_Score（分段映射到[0, 100]）
        # cq_score += calculate_cq_score_torch(outputs, targets)
        cut_targets,cut_outputs = truncate_batch_pairs(targets,outputs)
        val_loss_list = []
        cq_score_list = []
        for output_sample, target_sample in zip(cut_outputs, cut_targets):
            # 注意：这里需要确保 output_sample 和 target_sample 长度一致
            val_loss_sample = criterion(output_sample, target_sample)
            val_loss_list.append(val_loss_sample)
            cq_score_sample = calculate_cq_score_torch(output_sample, target_sample)
            cq_score_list.append(cq_score_sample)
                 
                # 当前 batch 的平均损失、平均分数
        batch_val_loss = torch.stack(val_loss_list).mean()
        batch_cq_score = torch.stack(cq_score_list).mean()

            # 累加到总和
        val_loss += batch_val_loss.item()
        cq_score += batch_cq_score.item()
        count += 1
        
# 最后做平均
    avg_val_loss = val_loss / count
    avg_cq_score = cq_score / count

print(f"Eval  | Val Loss: {avg_val_loss:.4f} | CQ_Score: {avg_cq_score:.4f}")
