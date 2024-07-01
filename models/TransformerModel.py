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
        self.no_gluc = no_gluc,
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'

        # EMBEDDING LINEAR LAYERS
        self.embedding_gluc = nn.Linear(1, self.num_features, dtype = self.dtype)
        self.embeddings = nn.ModuleList([nn.Linear(1, self.num_features, dtype = self.dtype) for _ in range(self.num_seqs)])
        
        # ENCODER LAYERS
        self.encoders = nn.ModuleList([nn.TransformerEncoderLayer(d_model=self.num_features, nhead=self.num_head, norm_first = self.norm_first, dtype = self.dtype) for _ in range(self.num_seqs)])

        # DECODER LAYERS
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.num_features, nhead=self.num_head, dtype = self.dtype)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, 1)
        # self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=1)  # Using a single layer

        # FULLY-CONNECTED LAYERS
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(self.num_features * self.num_seqs, self.num_features, dtype = self.dtype)
        self.fc2 = nn.Linear(self.num_features, self.num_features // 2, dtype = self.dtype)
        self.fc3 = nn.Linear(self.num_features // 2, 1, dtype = self.dtype)

        self.sos_token = 0

    # function: forward of model
    # input: src, tgt, tgt_mask
    # output: output after forward run through model
    def forward(self, tgt, src):
        batch_size, _, _ = src.shape
        if self.no_gluc:
            outputs = []
            idx = 0
            for layer in self.embeddings:
                feature = src[:, idx, :].unsqueeze(1).reshape(-1,1)
                output = layer(feature)
                output = output.view(batch_size, self.num_features, self.seq_length)
                outputs.append(output)
                idx += 1
            idx = 0
            for layer in self.encoders:
                output = outputs[idx].permute(0,2,1)
                outputs[idx] = layer(output)
                idx += 1

            out = torch.cat(outputs, -1).to(self.dtype)

            encoding = F.silu(self.fc1(out))
            
            tgt = tgt.unsqueeze(1).reshape(-1,1)
            tgt = self.embedding_gluc(tgt)
            tgt = tgt.view(batch_size, self.num_features, self.seq_length)
            tgt = tgt.permute(0,2,1)
            out = self.decoder(tgt = tgt, memory = encoding, tgt_mask = self.get_causal_mask(tgt.size(2)), tgt_is_causal=True)
            
            out = torch.tensor(out.clone().detach().requires_grad_(True), dtype=self.fc1.weight.dtype)
            # print(out)
            out = out.reshape(batch_size * self.seq_length, self.num_features)
            out = F.silu(self.fc2(self.dropout(out)))
            out = self.fc3(self.dropout(out))
            out = out.reshape(batch_size, 1, self.seq_length)
            return out
        # Src size must be (batch_size, src, sequence_length)
        # Tgt size must be (batch_size, tgt, sequence_length)
        outputs = []
        idx = 0
        for layer in self.embeddings:
            feature = src[:, idx, :].unsqueeze(1).reshape(-1,1)
            output = layer(feature)
            output = output.view(batch_size, self.num_features, self.seq_length)
            outputs.append(output)
            idx += 1
        idx = 0
        for layer in self.encoders:
            outputs[idx] = layer(outputs[idx])
            idx += 1

        out = torch.cat(outputs, -1).to(self.dtype)

        out = F.silu(self.fc1(out))

        tgt = tgt.unsqueeze(1).reshape(-1,1)
    
        tgt = self.embeddings[-1](tgt).unsqueeze(1)
        tgt = tgt.view(batch_size, self.num_features, self.seq_length)
            
        out = self.decoder(tgt = tgt, memory = out, tgt_mask = self.get_causal_mask(len(tgt)), tgt_is_causal=True)
        out = torch.tensor(out.clone().detach().requires_grad_(True), dtype=self.fc1.weight.dtype)
        out = out.reshape(batch_size * self.seq_length, self.num_features)
        print(out.argmax(dim = 2))
        out = F.silu(self.fc2(self.dropout(out)))
        out = self.fc3(self.dropout(out))
        out = out.reshape(batch_size, 1, self.seq_length)
        return out
    
    # function: creates a mask with 0's in bottom left of matrix
    # input: size
    # output: mask
    def get_tgt_mask(self, size) -> torch.tensor:
        mask = torch.tril(torch.ones(size,size) * float('-inf')).T
        for i in range(size):
            mask[i, i] = 0
        return mask.to(self.dtype).to(self.device)
    
    def get_causal_mask(self, size) -> torch.tensor:
        mask = torch.tril(torch.ones(size, size)).bool()
        return mask
