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

class TransformerModel(nn.Module):
    # Constructor
    def __init__(
        self,
        num_features,
        num_head,
        seq_length,
        dropout_p,
        norm_first,
        dtype,
        num_seqs,
        no_gluc = False):
        super(TransformerModel, self).__init__()

        # INFO
        self.model_type = "Transformer"
        self.num_features = num_features
        self.num_head = num_head
        self.seq_length = seq_length
        self.dropout_p = dropout_p
        self.norm_first = norm_first
        self.dtype = dtype
        self.num_seqs = num_seqs
        self.no_gluc = no_gluc

        # EMBEDDING LINEAR LAYERS
        self.embedding_gluc = nn.Linear(self.seq_length, self.num_features, dtype = self.dtype)
        self.embeddings = nn.ModuleList([nn.Linear(self.seq_length, self.num_features, dtype = self.dtype) for _ in range(self.num_seqs)])
        
        # ENCODER LAYERS
        self.encoders = nn.ModuleList([nn.TransformerEncoderLayer(d_model=self.num_features, nhead=self.num_head, norm_first = self.norm_first, dtype = self.dtype) for _ in range(self.num_seqs)])

        # DECODER LAYERS
        self.decoder = nn.TransformerDecoderLayer(d_model=self.num_features, nhead=self.num_head, dtype = self.dtype)
        # self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=1)  # Using a single layer

        # FULLY-CONNECTED LAYERS
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(self.num_features * self.num_seqs, self.num_features, dtype = self.dtype)
        self.fc2 = nn.Linear(self.num_features, self.num_features // 2, dtype = self.dtype)
        self.fc3 = nn.Linear(self.num_features // 2, self.seq_length, dtype = self.dtype)

    # function: forward of model
    # input: src, tgt, tgt_mask
    # output: output after forward run through model
    def forward(self, tgt, src):
        if self.no_gluc:
            outputs = []
            idx = 0
            for layer in self.embeddings:
                outputs.append(layer(src[:, idx, :]).unsqueeze(1))
                idx += 1
            idx = 0
            for layer in self.encoders:
                outputs[idx] = layer(outputs[idx])
                idx += 1

            out = torch.cat(outputs, -1).to(self.dtype)

            out = F.silu(self.fc1(out))
        
            tgt = self.embedding_gluc(tgt).unsqueeze(1)
            
            out = self.decoder(tgt = tgt, memory = out, tgt_mask = self.get_tgt_mask(len(tgt)))
            out = torch.tensor(out.clone().detach().requires_grad_(True), dtype=self.fc1.weight.dtype)
            out = F.silu(self.fc2(self.dropout(out)))
            out = self.fc3(self.dropout(out))
            return out
        # Src size must be (batch_size, src, sequence_length)
        # Tgt size must be (batch_size, tgt, sequence_length)
        outputs = []
        idx = 0
        for layer in self.embeddings:
            outputs.append(layer(src[:, idx, :]).unsqueeze(1))
            idx += 1
        idx = 0
        for layer in self.encoders:
            outputs[idx] = layer(outputs[idx])
            idx += 1

        out = torch.cat(outputs, -1).to(self.dtype)

        out = F.silu(self.fc1(out))
    
        tgt = self.embeddings[-1](tgt).unsqueeze(1)
        
        out = self.decoder(tgt = tgt, memory = out, tgt_mask = self.get_tgt_mask(len(tgt)))
        out = torch.tensor(out.clone().detach().requires_grad_(True), dtype=self.fc1.weight.dtype)
        out = F.silu(self.fc2(self.dropout(out)))
        out = self.fc3(self.dropout(out))
        return out
    
    # function: creates a mask with 0's in bottom left of matrix
    # input: size
    # output: mask
    def get_tgt_mask(self, size) -> torch.tensor:
        mask = torch.tril(torch.ones(size,size) * float('-inf')).T
        for i in range(size):
            mask[i, i] = 0
        return mask.to(self.dtype)
