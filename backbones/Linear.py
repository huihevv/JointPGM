import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.Linear_encoder = nn.Linear(self.seq_len, self.d_model)
        self.Linear_decoder = nn.Linear(self.d_model, self.pred_len)

    def Encoder(self, x):
        x = self.Linear_encoder(x.permute(0,2,1))
        return x 
    
    def Decoder(self, x):
        x = self.Linear_decoder(x).permute(0,2,1)
        return x 