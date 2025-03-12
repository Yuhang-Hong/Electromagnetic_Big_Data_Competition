# import os
# import pickle
# import numpy as np
# import pandas as pd

# # 设置数据所在的文件夹路径
# train_data_dir = "train_data"  # 请替换为实际路径

# # 存储所有数据的列表
# iq_data = []
# mod_codes = []
# mod_types = []
# symbol_widths = []
# true_label = []
# # 记录 IQ 数据和 mod_code 的最大长度，用于后续进行0填充

# # 1：BPSK，2：QPSK，3：8PSK，4：MSK，5：8QAM，6：16-QAM，7：32-QAM，8：8-APSK，9：16-APSK，10：32-APSK

# # 需要最后留一个true_label,即新建一列存储，需要与信号进行对应

# IQ_length = 0
# mode_code_type358 = 0  # 根据不同的mode_code的类型设置最大的mod_code的长度
# mode_code_type69 = 0  # 358代表调制类型的类别识别，下述同样
# mode_code_type710 = 0
# mode_code_type14 = 0
# mode_code_type2 = 0

# print("开始遍历调制类型文件夹...")

# for mod_type_folder in os.listdir(train_data_dir):
#     mod_type_folder_path = os.path.join(train_data_dir, mod_type_folder)
    
#     if os.path.isdir(mod_type_folder_path):
#         print(f"正在处理调制类型文件夹: {mod_type_folder}")
#         for csv_file in os.listdir(mod_type_folder_path):
#             csv_file_path = os.path.join(mod_type_folder_path, csv_file)
            
#             # 只处理 CSV 文件
#             if csv_file.endswith('.csv'):
#                 data = pd.read_csv(csv_file_path, header=None)
#                 # 假设 CSV 格式为：I, Q, mod_code, mod_type, symbol_width
#                 I_data = data[0].values
#                 Q_data = data[1].values
#                 iq_signal = np.array([I_data, Q_data])
                
#                 # 更新 IQ_length
#                 if IQ_length < iq_signal.shape[1]:
#                     IQ_length = iq_signal.shape[1]
                
#                 # 提取调制方式和码元宽度
#                 mod_type = int(data[3].values[0])
#                 symbol_width = data[4].values[0]
                
#                 mod_code = data[2].values
                
#                 # 根据不同 mod_type 更新对应的最大 mod_code 长度
#                 if mod_type in [3, 5, 8]:
#                     if mode_code_type358 < len(mod_code):
#                         mode_code_type358 = len(mod_code)
#                 elif mod_type in [6, 9]:
#                     if mode_code_type69 < len(mod_code):
#                         mode_code_type69 = len(mod_code)
#                 elif mod_type in [7, 10]:
#                     if mode_code_type710 < len(mod_code):
#                         mode_code_type710 = len(mod_code)
#                 elif mod_type in [1, 4]:
#                     if mode_code_type14 < len(mod_code):
#                         mode_code_type14 = len(mod_code)
#                 elif mod_type == 2:
#                     if mode_code_type2 < len(mod_code):
#                         mode_code_type2 = len(mod_code)
#                 else:
#                     print("mod_type 值错误")
                
#                 # 将数据添加到对应的列表中
#                 iq_data.append(iq_signal)
#                 mod_codes.append(mod_code)
#                 mod_types.append(mod_type)
#                 symbol_widths.append(symbol_width)
#                 true_label.append(mod_code)
                
#         print(f"调制类型文件夹：{mod_type_folder}处理完毕")
#         print("final IQ_length:", IQ_length)       
#         print("final mode_code_type358 length:", mode_code_type358)
#         print("final mode_code_type69 length:", mode_code_type69)
#         print("final mode_code_type710 length:", mode_code_type710)
#         print("final mode_code_type14 length:", mode_code_type14)
#         print("final mode_code_type2 length:", mode_code_type2)
        
# # ------------------------
# # 第一部分：对 IQ 数据做零填充
# # ------------------------
# padded_iq_data = []
# for iq_signal in iq_data:
#     current_length = iq_signal.shape[1]
#     if current_length < IQ_length:
#         # 对 axis=1 进行填充 (0,0) 表示行不变，(0, IQ_length - current_length) 表示列填充
#         padded_signal = np.pad(
#             iq_signal,
#             pad_width=((0, 0), (0, IQ_length - current_length)),
#             mode='constant',
#             constant_values=0
#         )
#     else:
#         padded_signal = iq_signal
#     padded_iq_data.append(padded_signal)

# # 将填充后的 IQ 数据转换为 NumPy 数组
# iq_data = np.array(padded_iq_data)  # 形状: (样本数, 2, IQ_length)

# # ------------------------
# # 第二部分：对 mod_code 做零填充,根据不同的调制类型，构建不同长度的码序列标签，其中依据为上述得到的mode_code_type358等长度
# # ------------------------

# padded_mod_codes = []
# for code, mtype in zip(mod_codes, mod_types):
#     # 根据调制类型选择对应的最大长度
#     if mtype in [3, 5, 8]:
#         desired_length = mode_code_type358
#     elif mtype in [6, 9]:
#         desired_length = mode_code_type69
#     elif mtype in [7, 10]:
#         desired_length = mode_code_type710
#     elif mtype in [1, 4]:
#         desired_length = mode_code_type14
#     elif mtype == 2:
#         desired_length = mode_code_type2
#     else:
#         print("mod_type 值错误")
#         desired_length = len(code)

#     current_length = len(code)
#     if current_length < desired_length:
#         # 对一维序列进行填充
#         padded_code = np.pad(
#             code,
#             pad_width=(0, desired_length - current_length),
#             mode='constant',
#             constant_values=0
#         )
#     else:
#         padded_code = code
#     padded_mod_codes.append(padded_code)

# # 如果需要，也可以将 padded_mod_codes 转换为 NumPy 数组
# mod_codes = np.array(padded_mod_codes)

# # 将调制方式和码元宽度转换为 NumPy 数组（它们本身是一维的）
# mod_types = np.array(mod_types)
# symbol_widths = np.array(symbol_widths) 
# true_label = np.array(true_label)
# # ------------------------
# # 打包并保存
# # ------------------------
# data_dict = {
#     'iq_data': iq_data,
#     'mod_codes': mod_codes,
#     'mod_types': mod_types,
#     'symbol_widths': symbol_widths,
#     'true_label': true_label
# }

# with open("data.pkl", "wb") as f:
#     pickle.dump(data_dict, f)

# print("数据已保存到 'data.pkl' 文件中！")

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
true_label = []

# 记录 IQ 数据和 mod_code 的最大长度，用于后续进行0填充
IQ_length = 0
# 针对不同调制方式的最大码序列长度
mode_code_type358 = 0   # [3, 5, 8]
mode_code_type69 = 0    # [6, 9]
mode_code_type710 = 0   # [7, 10]
mode_code_type14 = 0    # [1, 4]
mode_code_type2 = 0     # [2]

print("开始遍历调制类型文件夹...")

for mod_type_folder in os.listdir(train_data_dir):
    mod_type_folder_path = os.path.join(train_data_dir, mod_type_folder)
    
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
                
                # 提取调制方式和码元宽度
                mod_type = int(data[3].values[0])
                symbol_width = data[4].values[0]
                
                # 提取 mod_code
                mod_code = data[2].values  # 一维数组

                # 根据不同 mod_type 更新对应的最大 mod_code 长度
                if mod_type in [3, 5, 8]:
                    if mode_code_type358 < len(mod_code):
                        mode_code_type358 = len(mod_code)
                elif mod_type in [6, 9]:
                    if mode_code_type69 < len(mod_code):
                        mode_code_type69 = len(mod_code)
                elif mod_type in [7, 10]:
                    if mode_code_type710 < len(mod_code):
                        mode_code_type710 = len(mod_code)
                elif mod_type in [1, 4]:
                    if mode_code_type14 < len(mod_code):
                        mode_code_type14 = len(mod_code)
                elif mod_type == 2:
                    if mode_code_type2 < len(mod_code):
                        mode_code_type2 = len(mod_code)
                else:
                    print("mod_type 值错误:", mod_type)
                
                # 将数据添加到对应的列表中
                iq_data.append(iq_signal)
                mod_codes.append(mod_code)
                mod_types.append(mod_type)
                symbol_widths.append(symbol_width)
                true_label.append(mod_code)
                
        print(f"调制类型文件夹：{mod_type_folder}处理完毕")

print("----------- 统计结果 -----------")
print("final IQ_length:", IQ_length)
print("final mode_code_type358 length:", mode_code_type358)
print("final mode_code_type69 length:", mode_code_type69)
print("final mode_code_type710 length:", mode_code_type710)
print("final mode_code_type14 length:", mode_code_type14)
print("final mode_code_type2 length:", mode_code_type2)
print("--------------------------------")

# ------------------------
# 1) 对 IQ 数据做零填充
# ------------------------
padded_iq_data = []
for iq_signal in iq_data:
    current_length = iq_signal.shape[1]
    if current_length < IQ_length:
        padded_signal = np.pad(
            iq_signal,
            pad_width=((0, 0), (0, IQ_length - current_length)),
            mode='constant',
            constant_values=0
        )
    else:
        padded_signal = iq_signal
    padded_iq_data.append(padded_signal)

iq_data = np.array(padded_iq_data)  # 形状: (样本数, 2, IQ_length)

# ------------------------
# 2) 根据不同调制方式，对 mod_code 做零填充
# ------------------------
padded_mod_codes = []
for code, mtype in zip(mod_codes, mod_types):
    if mtype in [3, 5, 8]:
        desired_length = mode_code_type358
    elif mtype in [6, 9]:
        desired_length = mode_code_type69
    elif mtype in [7, 10]:
        desired_length = mode_code_type710
    elif mtype in [1, 4]:
        desired_length = mode_code_type14
    elif mtype == 2:
        desired_length = mode_code_type2
    else:
        # 如果出现异常 mtype，可以自行处理
        desired_length = len(code)

    current_length = len(code)
    if current_length < desired_length:
        padded_code = np.pad(
            code,
            pad_width=(0, desired_length - current_length),
            mode='constant',
            constant_values=0
        )
    else:
        padded_code = code
    padded_mod_codes.append(padded_code)

# mod_codes = np.array(padded_mod_codes, dtype=object)  
# 注意：如果每种调制方式的 desired_length 不同，则 mod_codes 会是一个object数组；
# 如果所有文件最终长度都相同，则可以转成规则的二维数组。

mod_types = np.array(mod_types)
symbol_widths = np.array(symbol_widths)
# true_label = np.array(true_label)

# ------------------------
# 3) 统计“最大长度”出现的次数
# ------------------------
count_358 = 0
count_69 = 0
count_710 = 0
count_14 = 0
count_2 = 0

for code, mtype in zip(mod_codes, mod_types):
    # code 依然是一维，只是已经填充
    # len(code) 会是填充后的长度
    if mtype in [3, 5, 8] and len(code) == mode_code_type358:
        count_358 += 1
    elif mtype in [6, 9] and len(code) == mode_code_type69:
        count_69 += 1
    elif mtype in [7, 10] and len(code) == mode_code_type710:
        count_710 += 1
    elif mtype in [1, 4] and len(code) == mode_code_type14:
        count_14 += 1
    elif mtype == 2 and len(code) == mode_code_type2:
        count_2 += 1

print("----------- 最大长度出现次数 -----------")
print(f"mode_code_type358={mode_code_type358} 的文件数:", count_358)
print(f"mode_code_type69={mode_code_type69} 的文件数:", count_69)
print(f"mode_code_type710={mode_code_type710} 的文件数:", count_710)
print(f"mode_code_type14={mode_code_type14} 的文件数:", count_14)
print(f"mode_code_type2={mode_code_type2} 的文件数:", count_2)
print("---------------------------------------")

# ------------------------
# 4) 打包并保存
# ------------------------
data_dict = {
    'iq_data': iq_data,
    'mod_codes': mod_codes,
    'mod_types': mod_types,
    'symbol_widths': symbol_widths,
    'true_label': true_label
}

with open("data.pkl", "wb") as f:
    pickle.dump(data_dict, f)

print("数据已保存到 'data.pkl' 文件中！")
