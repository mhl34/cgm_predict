import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetEncoder(nn.Module):
    def __init__(self, num_features, normalize = False, seq_len = 28):
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
        
    def forward(self, x):
        x = torch.tensor(x.clone().detach().requires_grad_(True), dtype=self.conv1.weight.dtype)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out
        
