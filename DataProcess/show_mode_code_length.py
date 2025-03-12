import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# 设置数据所在的文件夹路径（请替换为实际路径）
train_data_dir = "train_data"

# 用于存储每个 CSV 文件中 mod_code 的长度
mod_code_lengths = []

# 遍历每个调制类型文件夹及其 CSV 文件
for mod_type_folder in os.listdir(train_data_dir):
    mod_type_folder_path = os.path.join(train_data_dir, mod_type_folder)
    if os.path.isdir(mod_type_folder_path):
        for csv_file in os.listdir(mod_type_folder_path):
            if csv_file.endswith('.csv'):
                csv_file_path = os.path.join(mod_type_folder_path, csv_file)
                # 假设 CSV 格式为：I, Q, mod_code, mod_type, symbol_width
                data = pd.read_csv(csv_file_path, header=None)
                # mod_code 存在第三列，获取该列数据，并统计长度
                # print(data.shape)
                mod_code = data[2].dropna()
                # print("mode_code_length:",len(mod_code))
                mod_code_lengths.append(len(mod_code))

# 将结果转换为 numpy 数组，便于计算统计量
mod_code_lengths = np.array(mod_code_lengths)

# 输出统计信息
print("样本总数:", len(mod_code_lengths))
print("最小长度:", mod_code_lengths.min())
print("最大长度:", mod_code_lengths.max())
print("平均长度:", mod_code_lengths.mean())
print("中位数:", np.median(mod_code_lengths))

# 绘制直方图
plt.figure(figsize=(10, 6))
# 设置 bins 参数，使每个长度对应一个柱子
bins = range(mod_code_lengths.min(), mod_code_lengths.max() + 2)
plt.hist(mod_code_lengths, bins=bins, edgecolor='black', alpha=0.75)
plt.xlabel("Mode Code length")
plt.ylabel("counts")
plt.title("Mode Code length histogram")
plt.grid(True)
plt.savefig("mode_code_length_histogram.png")

# data = pd.read_csv(r'train_data/8APSK/data_1.csv',header=None)
# mod_code = data[2].dropna()

# print(data)
# print(mod_code)
# print(mod_code.shape)