import torch
import echotorch
import echotorch.nn as etnn
import numpy as np

# Define ESN model
class ESN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dtype, connectivity = 0.1, spectral_radius = 1.3, minimum_edges = 10, input_scaling = 0.6, bias_scaling = 1.0):
        super(ESN, self).__init__()
        self.dtype = dtype
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Internal matrix
        self.w_generator = echotorch.utils.matrix_generation.NormalMatrixGenerator(
            connectivity=connectivity,
            spetral_radius=spectral_radius,
            minimum_edges=minimum_edges
        )

        # Input weights
        self.win_generator = echotorch.utils.matrix_generation.NormalMatrixGenerator(
            connectivity=connectivity,
            scale=input_scaling,
            apply_spectral_radius=False,
            minimum_edges=minimum_edges
        )

        # Bias vector
        self.wbias_generator = echotorch.utils.matrix_generation.NormalMatrixGenerator(
            connectivity=connectivity,
            scale=bias_scaling,
            apply_spectral_radius=False
        )
        # Initialize the ESN with appropriate generators
        self.esn = etnn.ESN(input_dim = self.input_dim, hidden_dim = self.hidden_dim, output_dim = self.output_dim, 
                            w_generator=self.w_generator, win_generator=self.win_generator, wbias_generator=self.wbias_generator, dtype = self.dtype)

    def forward(self, x):
        return self.esn(x)