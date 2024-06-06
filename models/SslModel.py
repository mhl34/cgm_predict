import torch
import torch.nn as nn
import torch.nn.functional as F 
import random
from models.ChannelFC import ChannelFC

class SslModel(nn.Module):
    def __init__(self, mask_len, dropout = 0, seq_len = 28):
        super(SslModel, self).__init__()
        self.seq_len = seq_len
        self.num_seqs = 4
        # input: 28 x 3
        # 3 channels for each of the different modalities
        # output: 12 x 8
        self.conv1 = nn.Conv1d(in_channels = self.num_seqs, out_channels = 8, kernel_size = 5, stride = 2)
        # input: 12 x 8
        # output: 10 x 16
        self.conv2 = nn.Conv1d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 1)
        # input: 10 x 16
        # output: 6 x 64
        self.conv3 = nn.Conv1d(in_channels = 16, out_channels = 64, kernel_size = 5, stride = 1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64 * (((self.seq_len - 4) // 2) - 2 - 4), 64)
        self.fc2 = nn.Linear(64, 1)
        self.mask_len = mask_len

        self.decoder = nn.Sequential(
            ChannelFC(64, 6),
            nn.ConvTranspose1d(in_channels = 64, out_channels = 16, kernel_size = 5, stride = 1),
            nn.ConvTranspose1d(in_channels = 16, out_channels = 8, kernel_size = 3, stride = 1),
            nn.ConvTranspose1d(in_channels = 8, out_channels = self.num_seqs, kernel_size = 6, stride = 2)
        )

    
    def forward(self, x):
        x = torch.tensor(x.clone().detach().requires_grad_(True), dtype=self.conv1.weight.dtype)
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        masked_out = None
        if self.training:
            masked_x = self.getMasked(x, mask_len = self.mask_len)
            masked_out = F.relu(self.conv1(masked_x))
            masked_out = F.relu(self.conv2(masked_out))
            masked_out = F.relu(self.conv3(masked_out))
            masked_out = self.decoder(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(self.dropout(out)))
        out = F.relu(self.fc2(self.dropout(out)))
        return masked_out, out
    
    def getMasked(self, data, mask_len = 5):
        mask = torch.ones_like(data)
        _, _, seq_len = mask.shape
        index = random.randint(0, seq_len - mask_len - 1)
        mask[:,:,index:index + mask_len] = 0
        data = data * mask
        return data
