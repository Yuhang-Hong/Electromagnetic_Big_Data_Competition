import os
import pandas as pd

# 设置数据所在的文件夹路径（请替换为实际路径）
train_data_dir = "train_data"

# 用于存储每个码序列长度对应的样本数量
code_length_counts = {}

# 遍历每个调制类型文件夹及其中的 CSV 文件
for mod_type_folder in os.listdir(train_data_dir):
    mod_type_folder_path = os.path.join(train_data_dir, mod_type_folder)
    if os.path.isdir(mod_type_folder_path):
        for csv_file in os.listdir(mod_type_folder_path):
            if csv_file.endswith('.csv'):
                csv_file_path = os.path.join(mod_type_folder_path, csv_file)
                # 假设 CSV 格式为：I, Q, mod_code, mod_type, symbol_width
                try:
                    data = pd.read_csv(csv_file_path, header=None)
                except Exception as e:
                    print(f"读取文件 {csv_file_path} 出现错误：{e}")
                    continue

                # 获取 mod_code 数据并统计其长度
                mod_code = data[2].dropna()
                code_length = len(mod_code)
                # 更新统计字典
                code_length_counts[code_length] = code_length_counts.get(code_length, 0) + 1

# 对统计结果按码序列长度排序（升序）
sorted_pairs = sorted(code_length_counts.items(), key=lambda x: x[0])

# 将结果保存到 txt 文件中，每行格式：<码序列长度><制表符><样本数量>
output_filename = "mod_code_length_counts.txt"
with open(output_filename, "w", encoding="utf-8") as f:
    for length, count in sorted_pairs:
        f.write(f"{length}\t{count}\n")

print(f"结果已保存到 {output_filename}")

# import os
# import pandas as pd

# # 设置数据所在的文件夹路径（请替换为实际路径）
# train_data_dir = "train_data"

# # 用于存储每个码序列长度对应的样本数量
# code_length_counts = {}

# # 遍历每个调制类型文件夹及其中的 CSV 文件
# for mod_type_folder in os.listdir(train_data_dir):
#     mod_type_folder_path = os.path.join(train_data_dir, mod_type_folder)
#     if os.path.isdir(mod_type_folder_path):
#         for csv_file in os.listdir(mod_type_folder_path):
#             if csv_file.endswith('.csv'):
#                 csv_file_path = os.path.join(mod_type_folder_path, csv_file)
#                 # 假设 CSV 格式为：I, Q, mod_code, mod_type, symbol_width
#                 try:
#                     data = pd.read_csv(csv_file_path, header=None)
#                 except Exception as e:
#                     print(f"读取文件 {csv_file_path} 出现错误：{e}")
#                     continue

#                 # 获取 mod_code 数据并统计其长度
#                 mod_code = data[2].dropna()
#                 code_length = len(mod_code)
#                 # 更新统计字典
#                 code_length_counts[code_length] = code_length_counts.get(code_length, 0) + 1

# # 对统计结果按样本数量排序（降序排列）
# sorted_pairs = sorted(code_length_counts.items(), key=lambda x: x[1], reverse=True)

# # 将结果保存到 txt 文件中，每行格式：<码序列长度><制表符><样本数量>
# output_filename = "mod_code_length_counts_by_sample_count.txt"
# with open(output_filename, "w", encoding="utf-8") as f:
#     for length, count in sorted_pairs:
#         f.write(f"{length}\t{count}\n")

# print(f"结果已保存到 {output_filename}")
