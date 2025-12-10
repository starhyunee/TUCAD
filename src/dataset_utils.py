import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm



# === Dataset Utilities === #
class TimeSeriesWindowDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]

def create_sliding_windows(data, window_size=100, stride=1):
    time_len, dim = data.shape
    num_windows = (time_len - window_size) // stride + 1
    windows = [data[i:i+window_size] for i in range(0, num_windows * stride, stride)]
    return torch.stack(windows, dim=0)


# len(anomaly_score) == len(label)
def create_sliding_windows2(data, window_size= 100):
    windows = []
    T, _ = data.shape
    for i in range(T):
        if i > window_size:                 
            w = data[i - window_size + 1 : i + 1]
        elif i == window_size:             
            w = data[1 : window_size + 1]  
        else:                              
            pad_len = window_size - i
            pad = data[0].unsqueeze(0).repeat(pad_len, 1)  
            w = torch.cat([pad, data[1 : i + 1]], dim=0)  

        windows.append(w)

    return torch.stack(windows, dim=0)