import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loss import CosineSimilarityLoss
import pickle
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm 
import os 
from function import *
from model.IQSeqCompressor import IQSeqCompressor

# 读取存储的数据字典
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

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练集样本数：{len(train_loader.dataset)}")
print(f"测试集样本数：{len(val_loader.dataset)}")
# for i, (inputs, labels, mod) in enumerate(train_loader):
#     print(f"第{i+1}批次数据，输入形状：{inputs.shape}, 标签形状：{labels.shape}, mod_type 形状：{mod.shape}")
#     break


# 初始化模型
model = IQSeqCompressor().cuda()

criterion = CosineSimilarityLoss()

# 优化器（LAMB优化器更适合大batch）
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)

# 学习率预热调度
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda epoch: min(1.0, (epoch+1)/10)  # 前10epoch预热
)



# model = nn.Sequential(IQNormalize(), model).cuda()

best_score = float('inf')

for epoch in range(300):
    model.train()
    total_loss = 0.0
    
    for inputs, targets in tqdm(train_loader):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.float().cuda(non_blocking=True)  # 确保标签为浮点
        
        # 混合精度训练
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            
            cut_targets,cut_outputs = truncate_batch_pairs(targets,outputs)
            loss_list = []
            for output_sample, target_sample in zip(cut_outputs, cut_targets):
                # 注意：这里需要确保 output_sample 和 target_sample 长度一致
                loss_sample = criterion(output_sample, target_sample)
                loss_list.append(loss_sample)
                
            loss = torch.stack(loss_list).mean()
            # loss = criterion(outputs, targets)
        
        # 梯度更新
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    cq_score = 0.0
    count=0
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader):
            inputs = inputs.cuda()
            targets = targets.float().cuda()
            
            outputs = model(inputs, inference=True)
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
    
    # 定义保存模型的文件夹路径
    save_folder = "without_normlization"  # 可根据需要修改文件夹名称
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # 构建日志信息
    log_message = f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | CQ_Score: {avg_cq_score:.4f}"

    # 打印日志信息
    print(log_message)

    # 定义日志文件的完整路径
    log_file = os.path.join(save_folder, "log.txt")

    # 将日志信息追加写入到日志文件中
    with open(log_file, "a") as f:
        f.write(log_message + "\n")
   
    

    # 保存最佳模型
    if avg_val_loss < best_score:
        best_score = avg_val_loss
        save_path = os.path.join(save_folder, f"best_model_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)

