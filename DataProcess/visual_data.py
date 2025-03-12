import pickle
import numpy as np

with open("/root/autodl-tmp/data.pkl", "rb") as f:
    data_dict = pickle.load(f)

# 取出各部分数据
iq_data = data_dict['iq_data']           # shape: (N, 2, IQ_length)
mod_codes = data_dict['mod_codes']       # shape: (N, mod_codes_length)
mod_types = data_dict['mod_types']       # shape: (N,)
symbol_widths = data_dict['symbol_widths']  # shape: (N,)
print(iq_data.shape)
print(mod_codes.shape)
print(mod_types.shape)
print(symbol_widths.shape)
print(mod_codes[117777])
print(mod_types)
