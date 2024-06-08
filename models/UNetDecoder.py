import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetDecoder(nn.Module):
    def __init__(self, num_features, normalize = False, seq_len = 28, dropout = 0):
        super(UNetDecoder, self).__init__()
        self.upconv1 =nn.ConvTranspose1d(in_channels = 512, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        # output: 6 x 128
        # output: 12 x 128
        self.upconv2 = nn.ConvTranspose1d(in_channels = 128, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)
        # output: 12 x 128
        # output: 12 x 64
        self.upconv3 = nn.ConvTranspose1d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        # output: 12 x 64
        # output: 28 x 16
        self.upconv4 = nn.ConvTranspose1d(in_channels = 64, out_channels = 16, kernel_size = 4, stride = 2)
        # output: 28 x 16
        # output: 28 x 1
        self.upconv5 = nn.ConvTranspose1d(in_channels = 16, out_channels = 1, kernel_size = 3, stride = 1)
        self.dropout1d = nn.Dropout1d(dropout)
        
    def forward(self, x):
        x = torch.tensor(x.clone().detach().requires_grad_(True), dtype=self.upconv1.weight.dtype)
        out = F.silu(self.upconv1(self.dropout1d(x)))
        out = F.silu(self.upconv2(self.dropout1d(out)))
        out = F.silu(self.upconv3(self.dropout1d(out)))
        out = F.silu(self.upconv4(self.dropout1d(out)))
        out = self.upconv5(out)
        return out
