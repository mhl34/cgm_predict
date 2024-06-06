import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetDecoder(nn.Module):
    def __init__(self, num_features, normalize = False, seq_len = 28):
        super(UNetDecoder, self).__init__()
        # output: 6 x 128
        # output: 12 x 128
        self.upconv1 = nn.ConvTranspose1d(in_channels = 128, out_channels = 128, kernel_size = 4, stride = 2, padding = 1)
        # output: 12 x 128
        # output: 12 x 64
        self.upconv2 = nn.ConvTranspose1d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        # output: 12 x 64
        # output: 28 x 16
        self.upconv3 = nn.ConvTranspose1d(in_channels = 64, out_channels = 16, kernel_size = 4, stride = 2)
        # output: 28 x 16
        # output: 28 x 1
        self.upconv4 = nn.ConvTranspose1d(in_channels = 16, out_channels = 1, kernel_size = 3, stride = 1)
        
    def forward(self, x):
        x = torch.tensor(x.clone().detach().requires_grad_(True), dtype=self.upconv1.weight.dtype)
        out = self.upconv1(x)
        out = self.upconv2(out)
        out = self.upconv3(out)
        out = self.upconv4(out)
        return out
