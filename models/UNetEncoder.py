import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetEncoder(nn.Module):
    def __init__(self, num_features, normalize = False, seq_len = 28, dropout = 0):
        super(UNetEncoder, self).__init__()
        self.num_features = num_features
        # input: 28 x 4
        # 3 channels for each of the different modalities
        # output: 12 x 8
        self.conv1 = nn.Conv1d(in_channels = self.num_features, out_channels = 8, kernel_size = 5, stride = 2)
        # input: 12 x 8
        # output: 10 x 32
        self.conv2 = nn.Conv1d(in_channels = 8, out_channels = 32, kernel_size = 3, stride = 1)
        # input: 10 x 32
        # output: 6 x 128
        self.conv3 = nn.Conv1d(in_channels = 32, out_channels = 128, kernel_size = 5, stride = 1)
        # input: 6 x 128
        # input 6 x 512
        self.conv4 = nn.Conv1d(in_channels = 128, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.dropout1d = nn.Dropout1d(dropout)
        
    def forward(self, x):
        x = torch.tensor(x.clone().detach().requires_grad_(True), dtype=self.conv1.weight.dtype)
        out = F.silu(self.conv1(x))
        out = F.silu(self.conv2(self.dropout1d(out)))
        out = F.silu(self.conv3(self.dropout1d(out)))
        out = F.silu(self.conv4(self.dropout1d(out)))
        return out
        
