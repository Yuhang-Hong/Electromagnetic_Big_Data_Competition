# import os
# import pickle

# import numpy as np
# import pandas as pd

# # 设置数据所在的文件夹路径
# train_data_dir = "dataset"  # 请替换为实际路径

# # 存储所有数据的列表
# iq_data = []
# mod_codes = []
# mod_types = []
# symbol_widths = []

# IQ_length = 0
# mod_codes_length = 0
# # with open("data.pkl", "rb") as f:
# #     loaded_object = pickle.load(f)

# # 遍历每个调制类型的文件夹
# print("开始遍历调制类型文件夹...")
# for mod_type_folder in os.listdir(train_data_dir):
#     mod_type_folder_path = os.path.join(train_data_dir, mod_type_folder)

#     # 如果是文件夹，继续遍历里面的 CSV 文件
#     if os.path.isdir(mod_type_folder_path):
#         print(f"正在处理调制类型文件夹: {mod_type_folder}")
#         for csv_file in os.listdir(mod_type_folder_path):
#             csv_file_path = os.path.join(mod_type_folder_path, csv_file)

#             # 只处理 CSV 文件
#             if csv_file.endswith('.csv'):
#                 # print(f"正在处理文件: {csv_file}")
#                 # 读取 CSV 文件，假设没有标题行，列为 I、Q、mod_code、mod_type、symbol_width
#                 data = pd.read_csv(csv_file_path, header=None)

#                 # 提取 I 和 Q 数据列
#                 I_data = data[0].values
#                 Q_data = data[1].values
#                 iq_signal = np.array([I_data, Q_data])

#                 if IQ_length < len(I_data):
#                     IQ_length = len(I_data)
                
#                 # 提取调制码序列、调制方式和码元宽度
#                 mod_code = data[2].values
                
#                 if mod_codes_length < len(mod_code):
#                     mod_codes_length = len(mod_code)
                    
#                 mod_type = data[3].values[0]  # 只有一个调制方式
#                 symbol_width = data[4].values[0]  # 只有一个码元宽度

#                 # 将数据添加到对应的列表中
#                 iq_data.append(iq_signal)
#                 mod_codes.append(mod_code)
#                 mod_types.append(mod_type)
#                 symbol_widths.append(symbol_width)

# padded_data = []
# # 遍历每个信号，进行零填充
# for iq_signal in iq_data:
#     if iq_signal.shape[1] < IQ_length:
#         # 使用 np.pad() 填充零，(0, max_length - len(iq_signal)) 表示将信号填充到 max_length 长度
#         padded_signal = np.pad(iq_signal, ((0, 0), (0, IQ_length - iq_signal.shape[1])), mode='constant', constant_values=0)
#     else:
#         padded_signal = iq_signal  # 如果信号已经达到最大长度，则不进行填充
#     padded_data.append(padded_signal)
    
# for mod_code in mod_codes:
#     if mod_code.shape[1] < mod_codes_length:
#         padded_mod_code = np.pad(mod_code, 0, )    
    

# padded_code = []

# # 将填充后的数据转换为 NumPy 数组
# iq_data = np.array(padded_data)

# # 将列表转换为 NumPy 数组
# print("转换数据为 NumPy 数组...")
# iq_data = np.array(iq_data)
# # mod_codes = np.array(mod_codes)
# mod_types = np.array(mod_types)
# symbol_widths = np.array(symbol_widths)

# # 合并数据保存为一个字典（或者直接保存为单独的数组）
# print("准备保存数据到 'data.npy' 文件...")
# data = {
#     'iq_data': iq_data,
#     'mod_codes': mod_codes,
#     'mod_types': mod_types,
#     'symbol_widths': symbol_widths
# }

# # 保存为 pkl 文件
# with open("data.pkl", "wb") as f:
#     pickle.dump(data, f)

# print("数据已保存到 'data.npy' 文件中")


import os
import pickle
import numpy as np
import pandas as pd

# 设置数据所在的文件夹路径
train_data_dir = "train_data"  # 请替换为实际路径

# 存储所有数据的列表
iq_data = []
mod_codes = []
mod_types = []
symbol_widths = []

# 记录 IQ 数据和 mod_code 的最大长度，用于后续填充
IQ_length = 0
mod_codes_length = 0

print("开始遍历调制类型文件夹...")
for mod_type_folder in os.listdir(train_data_dir):
    mod_type_folder_path = os.path.join(train_data_dir, mod_type_folder)

    # 如果是文件夹，继续遍历里面的 CSV 文件
    if os.path.isdir(mod_type_folder_path):
        print(f"正在处理调制类型文件夹: {mod_type_folder}")
        for csv_file in os.listdir(mod_type_folder_path):
            csv_file_path = os.path.join(mod_type_folder_path, csv_file)

            # 只处理 CSV 文件
            if csv_file.endswith('.csv'):
                data = pd.read_csv(csv_file_path, header=None)
                # 假设 CSV 格式为：I, Q, mod_code, mod_type, symbol_width
                I_data = data[0].values
                Q_data = data[1].values
                iq_signal = np.array([I_data, Q_data])
                
                # 更新 IQ_length
                if IQ_length < iq_signal.shape[1]:
                    IQ_length = iq_signal.shape[1]
                
                # 提取调制码序列
                mod_code = data[2].dropna()
                # print("current MOD_CODE length:", len(mod_code))
                # 更新 mod_codes_length
                if mod_codes_length < len(mod_code):
                    mod_codes_length = len(mod_code)
                    
                # print("current mod_codes_length:", mod_codes_length)
                # 提取调制方式和码元宽度
                mod_type = data[3].values[0]
                symbol_width = data[4].values[0]

                # 将数据添加到对应的列表中
                iq_data.append(iq_signal)
                mod_codes.append(mod_code)
                mod_types.append(mod_type)
                symbol_widths.append(symbol_width)
        print(f"调制类型文件夹：{mod_type_folder}处理完毕")
        print("final mod_codes length:", mod_codes_length)
        print("final IQ_length:", IQ_length)
# ------------------------
# 第一部分：对 IQ 数据做零填充
# ------------------------
padded_iq_data = []
for iq_signal in iq_data:
    current_length = iq_signal.shape[1]
    if current_length < IQ_length:
        # 对 axis=1 进行填充 (0,0) 表示行不变，(0, IQ_length - current_length) 表示列填充
        padded_signal = np.pad(
            iq_signal,
            pad_width=((0, 0), (0, IQ_length - current_length)),
            mode='constant',
            constant_values=0
        )
    else:
        padded_signal = iq_signal
    padded_iq_data.append(padded_signal)

# 将填充后的 IQ 数据转换为 NumPy 数组
iq_data = np.array(padded_iq_data)  # 形状: (样本数, 2, IQ_length)

# ------------------------
# 第二部分：对 mod_code 做零填充
# ------------------------
padded_mod_codes = []
for code in mod_codes:
    current_length = len(code)
    if current_length < mod_codes_length:
        # 对一维序列进行填充
        padded_code = np.pad(
            code,
            pad_width=(0, mod_codes_length - current_length),
            mode='constant',
            constant_values=0
        )
    else:
        padded_code = code
    padded_mod_codes.append(padded_code)

# 将填充后的 mod_code 转换为 NumPy 数组
mod_codes = np.array(padded_mod_codes)  # 形状: (样本数, mod_codes_length)

# 将调制方式和码元宽度转换为 NumPy 数组（它们本身是一维的）
mod_types = np.array(mod_types)
symbol_widths = np.array(symbol_widths)

# ------------------------
# 打包并保存
# ------------------------
data_dict = {
    'iq_data': iq_data,
    'mod_codes': mod_codes,
    'mod_types': mod_types,
    'symbol_widths': symbol_widths
}

with open("data.pkl", "wb") as f:
    pickle.dump(data_dict, f)

print("数据已保存到 'data.pkl' 文件中！")
