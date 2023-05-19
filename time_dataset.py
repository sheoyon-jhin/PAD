import numpy as np
import torch
import sys


def normalize(data):
    # import pdb ; pdb.set_trace()
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data
#numerator array([ 244.100052,  250.780014,  239.402664,  244.543396,  244.543396,7922.      ])
#denominator array([2.55611992e+03, 2.54943996e+03, 2.52069743e+03, 2.54834650e+03,2.54834650e+03, 2.49701520e+07])
# 0 : output * (2.55611992e+03)+ (1e-7) + 244.100052
# 1 : output * (2.54943996e+03)+ (1e-7) + 250.780014
# 2 : output * (2.52069743e+03)+ (1e-7) + 239.402664
# 3 : output * (2.54834650e+03)+ (1e-7) + 244.543396
# 4 : output * (2.54834650e+03)+ (1e-7) + 244.543396
# 5 : output * (2.49701520e+07)+ (1e-7) + 7922
def to_tensor(data):
    return torch.from_numpy(data).float()


def batch_generator(dataset, batch_size):
    dataset_size = len(dataset)
    idx = torch.randperm(dataset_size)
    batch_idx = idx[:batch_size]
    batch = torch.stack([to_tensor(dataset[i]) for i in batch_idx])
    return batch


class TimeDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, seq_len):
        
        data = np.loadtxt(data_path, delimiter=",", skiprows=1)
        total_length = len(data)
        data = data[::-1]
        
        self.min_val = np.min(data, 0)
        self.max_val = np.max(data, 0) - np.min(data, 0)
        
        norm_data = normalize(data)
        total_length = len(norm_data)
        idx = np.array(range(total_length)).reshape(-1,1)
        # norm_data = np.concatenate((norm_data,idx),axis=1)#맨 뒤에 관측시간에 대한 정보 저장

        seq_x_data = []
        seq_y_data = []
        for i in range(len(norm_data) - seq_len + 1): # 1은 한칸씩 밀면서 보는 것 ! 즉 시간 0 ~ 24 값 보고 그다음 텀에서 시간 1 ~ 25 이렇게 봄 
            # 총 3662개의 sequence가 들어간다. 
            x = norm_data[i : i + seq_len - 1]
            y = norm_data[i + seq_len - 1]
            seq_x_data.append(x)
            seq_y_data.append(y)
        
        self.X_data = []
        self.Y_data = []

        idx = torch.randperm(len(seq_x_data))
        for i in range(len(seq_x_data)):
            self.X_data.append(torch.tensor(seq_x_data[idx[i]]))
            self.Y_data.append(torch.tensor(seq_y_data[idx[i]]))

    def __getitem__(self, index):
        return self.X_data[index], self.Y_data[index]

    def __len__(self):
        return len(self.samples)
