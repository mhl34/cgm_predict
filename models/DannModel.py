import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GradientReversalLayer import GradientReversalLayer

# input: representations, which depend on the type of model
# output: guess on domain (Person as Domain)
# params: modelType
class DannModel(nn.Module):
    def __init__(self, modelType, samples, dropout = 0.5, seq_len = 28):
        super(DannModel, self).__init__()
        self.seq_len = seq_len
        self.num_seqs = 4
        self.modelType = modelType
        self.samples = samples
        self.dropout_layer = nn.Dropout(p = dropout)
        self.featureExtractor = nn.Sequential(
            nn.Conv1d(in_channels = self.num_seqs, out_channels = 8, kernel_size = 5, stride = 2),
            nn.ReLU(),
            nn.Conv1d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.Conv1d(in_channels = 16, out_channels = 64, kernel_size = 5, stride = 1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            self.dropout_layer,
            nn.Linear(64 * (((self.seq_len - 4) // 2) - 2 - 4), 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.inputDim = self.inputDimCalc(self.modelType)
        self.adversary = nn.Sequential(
            self.dropout_layer,
            nn.Linear(self.inputDim, 64),
            nn.ReLU(),
            self.dropout_layer,
            nn.Linear(64, 32),
            nn.ReLU(),
            self.dropout_layer,
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, len(self.samples))
        )

    def forward(self, x, alpha):
        x = torch.tensor(x.clone().detach().requires_grad_(True), dtype=self.featureExtractor[0].weight.dtype)
        features = self.featureExtractor(x)
        features = features.view(features.size(0), -1)
        task_output = self.classifier(features)
        dann_output = None
        if self.training:
            features = GradientReversalLayer.apply(features, alpha)
            dann_output = self.adversary(features)
        return dann_output, task_output

    def inputDimCalc(self, modelType):

        if modelType == "conv1d" or modelType == "dann":
            return 64 * (((self.seq_len - 4) // 2) - 2 - 4)
        if modelType == "lstm":
            return 100 * 3
        if modelType == "transformer":
            return 1024 * 3
        return -1
    
    def swish(self, x):
        return x * torch.sigmoid(x)