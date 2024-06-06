from models.UNetDecoder import UNetDecoder
from models.UNetEncoder import UNetEncoder
from models.ChannelFC import ChannelFC
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, num_features, normalize = False, seq_len = 28, num_encoder_features = 128, num_encoder_channels = 6):
        super(UNet, self).__init__()
        self.num_features = num_features
        self.normalize = normalize
        self.seq_len = seq_len
        self.encoder = UNetEncoder(self.num_features, self.normalize, self.seq_len)
        self.decoder = UNetDecoder(self.num_features, self.normalize, self.seq_len)
        self.channel_fc = ChannelFC(num_encoder_features, num_encoder_channels)
        
    def forward(self, x):
        out = self.encoder(x)
        out = self.channel_fc(out)
        out = self.decoder(out)
        return out
