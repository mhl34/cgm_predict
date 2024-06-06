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
        num_seqs = 5,
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
        self.embedding_acc = nn.Linear(self.seq_length, self.num_features, dtype = self.dtype)
        self.embedding_sugar = nn.Linear(self.seq_length, self.num_features, dtype = self.dtype)
        self.embedding_carb = nn.Linear(self.seq_length, self.num_features, dtype = self.dtype)
        self.embedding_minutes = nn.Linear(self.seq_length, self.num_features, dtype = self.dtype)
        self.embedding_hba1c = nn.Linear(self.seq_length, self.num_features, dtype = self.dtype)
        self.embedding_gluc = nn.Linear(self.seq_length, self.num_features, dtype = self.dtype)

        # ENCODER LAYERS
        self.encoder_acc = nn.TransformerEncoderLayer(d_model=self.num_features, nhead=self.num_head, norm_first = self.norm_first, dtype = self.dtype)
        self.encoder_sugar = nn.TransformerEncoderLayer(d_model=self.num_features, nhead=self.num_head, norm_first = self.norm_first, dtype = self.dtype)
        self.encoder_carb = nn.TransformerEncoderLayer(d_model=self.num_features, nhead=self.num_head, norm_first = self.norm_first, dtype = self.dtype)
        self.encoder_minutes = nn.TransformerEncoderLayer(d_model=self.num_features, nhead=self.num_head, norm_first = self.norm_first, dtype = self.dtype)
        self.encoder_hba1c = nn.TransformerEncoderLayer(d_model=self.num_features, nhead=self.num_head, norm_first = self.norm_first, dtype = self.dtype)
        self.encoder_gluc_past = nn.TransformerEncoderLayer(d_model=self.num_features, nhead=self.num_head, norm_first = self.norm_first, dtype = self.dtype)

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
            sugar = self.embedding_sugar(src[:, 0, :]).unsqueeze(1)
            carb = self.embedding_carb(src[:, 1, :]).unsqueeze(1)
            minutes = self.embedding_minutes(src[:, 2, :]).unsqueeze(1)
            hba1c = self.embedding_hba1c(src[:, 3, :]).unsqueeze(1)

            sugarTransformerOut = self.encoder_sugar(sugar)
            carbTransformerOut = self.encoder_carb(carb)
            minutesTransformerOut = self.encoder_minutes(minutes)
            hba1cTransformerOut = self.encoder_hba1c(hba1c)

            out = torch.cat((sugarTransformerOut, carbTransformerOut, minutesTransformerOut, hba1cTransformerOut), -1).to(self.dtype)

            out = F.silu(self.fc1(out))
        
            tgt = self.embedding_gluc(tgt).unsqueeze(1)
            
            out = self.decoder(tgt = tgt, memory = out, tgt_mask = self.get_tgt_mask(len(tgt)))
            out = torch.tensor(out.clone().detach().requires_grad_(True), dtype=self.fc1.weight.dtype)
            out = F.silu(self.fc2(self.dropout(out)))
            out = self.fc3(self.dropout(out))
            return out
        # Src size must be (batch_size, src, sequence_length)
        # Tgt size must be (batch_size, tgt, sequence_length)
        # sugarMean, carbMean, minMean, hba1cMean, glucPastMean, glucMean
        # acc = self.embedding_acc(src[:, 0, :]).unsqueeze(1)
        sugar = self.embedding_sugar(src[:, 0, :]).unsqueeze(1)
        carb = self.embedding_carb(src[:, 1, :]).unsqueeze(1)
        minutes = self.embedding_minutes(src[:, 2, :]).unsqueeze(1)
        hba1c = self.embedding_hba1c(src[:, 3, :]).unsqueeze(1)
        gluc_past = self.embedding_gluc(src[:, 4, :]).unsqueeze(1)

        # accTransformerOut = self.encoder_acc(acc)
        sugarTransformerOut = self.encoder_sugar(sugar)
        carbTransformerOut = self.encoder_carb(carb)
        minutesTransformerOut = self.encoder_minutes(minutes)
        hba1cTransformerOut = self.encoder_hba1c(hba1c)
        glucPastTransformerOut = self.encoder_gluc_past(gluc_past)

        out = torch.cat((sugarTransformerOut, carbTransformerOut, minutesTransformerOut, hba1cTransformerOut, glucPastTransformerOut), -1).to(self.dtype)
        # out = torch.cat((accTransformerOut, sugarTransformerOut, carbTransformerOut, minutesTransformerOut, hba1cTransformerOut, glucPastTransformerOut), -1).to(self.dtype)
        # out = torch.cat((edaTransformerOut, hrTransformerOut, tempTransformerOut, accTransformerOut), -1).to(self.dtype)

        # out = F.relu(self.fc1(self.dropout(out)))
        out = F.silu(self.fc1(out))
        
        tgt = self.embedding_gluc(tgt).unsqueeze(1)
        
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
