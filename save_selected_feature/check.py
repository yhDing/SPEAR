import numpy as np
import torch
dataset = 'Cora'
val_32 = np.load(f'{dataset}/val_32.npy')
val_64 = np.load(f'{dataset}/val_64.npy')
print(val_32.shape)
max_index_32 = np.argsort(val_32)[::-1][:20]
max_index_64 = np.argsort(val_64)[::-1][:20]
print(type(max_index_32))
