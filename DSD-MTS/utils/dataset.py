import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset
from utils.time_features import get_time_features


class TSForecastDataset(Dataset):
    def __init__(self, data_path='dataset/traffic.npz', flag='train', size=(12,12), split=(0.1, 0.2), 
                 scale=True, scale_method="std", time_features='s', normalise_time_features=True):
        self.seq_len, self.pred_len = size[0], size[1]
        self.set_type = {'train': 0 ,'val': 1, 'test': 2}[flag]
        self.ratio_val, self.ratio_test = split[0], split[1]
        self.scale = scale
        self.scale_method = scale_method
        self.time_features = time_features
        self.normalise_time_features = normalise_time_features
        self.timestamps = None
        self.scaler = None
        self.__read_data__(data_path)

    def __read_data__(self, data_path):
        df_raw = pd.read_csv(data_path)
        df_raw = df_raw.dropna()
        
        cols = list(df_raw.columns)
        cols.remove('date')
        raw_data = df_raw[cols].values

        num_val, num_test = int(len(raw_data)*self.ratio_val), int(len(raw_data)*self.ratio_test)
        num_train = len(raw_data) - num_val - num_test
        border1s = [0, num_train, num_train+num_val]
        border2s = [num_train, num_train+num_val, len(raw_data)]
        left_border = border1s[self.set_type]
        right_border = border2s[self.set_type]

        # normalize
        if self.scale_method == "std":
            self.scaler = StandardScaler()
        elif self.scale_method == "min-max":
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            print("Normalization method error!")
        if self.scale:
            train_data = raw_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            raw_data = self.scaler.transform(raw_data)
        else:
            raw_data = raw_data

        self.data_x = raw_data[left_border:right_border]  # input
        self.data_y = raw_data[left_border:right_border]  # output
        self.timestamps = get_time_features(pd.to_datetime(df_raw.date[left_border:right_border].values), 
                                            normalise=self.normalise_time_features, features=self.time_features)
        self.N = self.data_x.shape[1]  # number of series
        self.train_data = raw_data[border1s[0]:border2s[0]]  # train data

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        x_time = self.timestamps[s_begin:s_end]
        y_time = self.timestamps[r_begin:r_end]
        return seq_x, seq_y, x_time, y_time
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, raw_data):
        return self.scaler.inverse_transform(raw_data)

        


