import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ChannelFC import ChannelFC
import random

class LstmModel(nn.Module):
    def __init__(self, num_features, input_size, hidden_size, num_layers = 8, batch_first = True, dropout_p = 0.5, dtype = torch.float64, seq_len = 28, bidirectional = False):
        super(LstmModel, self).__init__()
        self.num_features = num_features
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout_p = dropout_p
        self.dtype = dtype
        self.seq_len = seq_len
        self.bidirectional = bidirectional
        self.lstm_encoder = nn.LSTM(input_size = self.input_size * self.num_features, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first = self.batch_first, dropout = self.dropout_p, dtype = self.dtype, bidirectional = self.bidirectional)
        self.lstm_decoder = nn.LSTM(input_size = self.hidden_size, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first = self.batch_first, dropout = self.dropout_p, dtype = self.dtype)
        self.fc1 = nn.Linear(self.hidden_size * 2 if self.bidirectional else self.hidden_size, 64, dtype = self.dtype)
        self.fc2 = nn.Linear(64, 32, dtype = self.dtype)
        self.fc3 = nn.Linear(32, self.input_size, dtype = self.dtype)
        # self.ssl = True
        self.mask_len = 7
        self.dropout = nn.Dropout(self.dropout_p)
        self.decoder = nn.Identity()

    def forward(self, x):
        x = x.reshape(x.size(0), -1).to(self.dtype)
        out, _ = self.lstm_encoder(x)

        out, _ = self.lstm_decoder(out)       

        out = out.reshape(out.size(0), -1).to(self.dtype)

        out = F.silu(self.fc1(self.dropout(out)))
        out = F.silu(self.fc2(self.dropout(out)))
        out = self.fc3(self.dropout(out))

        return out
    
    def getMasked(self, data, mask_len = 5):
        mask = torch.ones_like(data)
        _, _, seq_len = mask.shape
        index = random.randint(0, seq_len - mask_len - 1)
        mask[:,:,index:index + mask_len] = 0
        data = data * mask
        return data
