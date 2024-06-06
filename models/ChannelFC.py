import torch
import torch.nn as nn

class ChannelFC(nn.Module):
    def __init__(self, num_features, num_channels):
        super(ChannelFC, self).__init__()
        self.num_channels = num_channels
        self.num_features = num_features
        self.fc_layers = nn.ModuleList([
            nn.Linear(num_channels, num_channels) for _ in range(num_features)
        ])

    def forward(self, x):
        batch_size, num_features, num_channels = x.size()
        x_reshaped = x.view(batch_size, self.num_features, self.num_channels)
        fc_outputs = [fc(x_reshaped[:,i,:].view(batch_size, -1)) for i, fc in enumerate(self.fc_layers)]
        fc_outputs = [i.view(batch_size, self.num_channels) for i in fc_outputs]
        output = torch.stack(fc_outputs, dim = 1)
        return output