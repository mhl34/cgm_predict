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
        self.hidden_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.lstm_encoder = nn.LSTM(input_size = self.seq_len, hidden_size = self.hidden_size, num_layers = self.num_layers, batch_first = self.batch_first, dropout = self.dropout_p, dtype = self.dtype, bidirectional = self.bidirectional)
        self.batch_norm1 = nn.BatchNorm1d(self.seq_len, dtype = self.dtype)
        self.batch_norm2 = nn.BatchNorm1d(self.hidden_size, dtype = self.dtype)
        # self.batch_norm = nn.BatchNorm1d(self.seq_len, dtype = self.dtype)
        self.fc1 = nn.Linear(self.hidden_size, 64, dtype = self.dtype)
        self.fc2 = nn.Linear(64, self.input_size, dtype = self.dtype)
        self.dropout = nn.Dropout(self.dropout_p)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x):
        dim1 = self.num_layers * 2 if self.bidirectional else self.num_layers
        h0 = torch.zeros(dim1, x.size(0), self.hidden_size).to(self.device).to(self.dtype)
        c0 = torch.zeros(dim1, x.size(0), self.hidden_size).to(self.device).to(self.dtype)
        batch_size, _, _ = x.shape
        x = x.reshape(batch_size * self.num_features, self.seq_len)
        x = self.batch_norm1(x)
        x = x.reshape(batch_size, self.num_features, self.seq_len)
        output, (hn, cn) = self.lstm_encoder(x, (h0, c0))
        hn = hn.permute(1,0,2)
        hn_out = hn[:, -1, :]
        hn_out = (hn_out - hn.mean()) / hn.std()
        out = F.silu(self.fc1(self.dropout(hn_out)))
        out = self.fc2(self.dropout(out))
        return out
    
    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.ndimension() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
