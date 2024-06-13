import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import math
import numpy as np
import random
import datetime
import sys
import os
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

class LstmEnhancedModel(nn.Module):
    # Constructor
    def __init__(
        self,
        hidden_size,
        num_layers,
        seq_length,
        dropout_p,
        norm_first,
        dtype,
        num_seqs,
        batch_first,
        bidirectional,
        no_gluc=False):
        super(LstmEnhancedModel, self).__init__()

        # INFO
        self.model_type = "Transformer"
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.dropout_p = dropout_p
        self.norm_first = norm_first
        self.dtype = dtype
        self.num_seqs = num_seqs
        self.no_gluc = no_gluc
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        # EMBEDDING LINEAR LAYERS
        self.embeddings = nn.ModuleList([nn.Linear(self.seq_length, self.seq_length, dtype=self.dtype) for _ in range(self.num_seqs)])

        # ENCODER LAYERS
        self.encoders = nn.ModuleList([nn.LSTM(input_size=1, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout_p, dtype=self.dtype, bidirectional=self.bidirectional) for _ in range(self.num_seqs)])

        # DECODER LAYERS
        self.decoder = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout_p, dtype=self.dtype, bidirectional=self.bidirectional)

        # FULLY-CONNECTED LAYERS
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(self.seq_length * self.num_seqs, self.seq_length, dtype=self.dtype)
        self.fc2 = nn.Linear(self.hidden_size, 64, dtype=self.dtype)
        self.fc3 = nn.Linear(64, self.seq_length, dtype=self.dtype)

        self.batch_norm = nn.BatchNorm1d(self.seq_length, dtype=self.dtype)

    # function: forward of model
    # input: src
    # output: output after forward run through model
    def forward(self, src):
        outputs = []
        idx = 0
        for layer in self.embeddings:
            outputs.append(layer(src[:, idx, :]).unsqueeze(1))
            idx += 1
        
        idx = 0
        for layer in self.encoders:
            model_input = outputs[idx].permute(0, 2, 1)
            encoder_output, _ = layer(model_input)
            encoder_output = encoder_output[:, :, -1].unsqueeze(1)
            outputs[idx] = self.batch_norm(encoder_output)
            idx += 1

        out = torch.cat(outputs, -1).to(self.dtype)

        out = F.silu(self.fc1(self.dropout(out)))
        out = F.silu(self.fc2(self.dropout(out)))
        out = self.fc3(self.dropout(out))
        return out