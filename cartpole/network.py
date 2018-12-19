from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Mlp(nn.Module):
    """
    Multi-layer perceptron implementation with pytorch
    
    Initialize the network with

    in: 

    input_dim [int]
    hidden_dim [array-like with ints]
    output_dim [int]
    """
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 output_dim):
        super(Mlp, self).__init__()
        # For efficient storage of hidden layers:
        self.hidden_layers = OrderedDict()

        # Define the input layer
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])

        # Define the hidden layers
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers['h_{}'.format(i)] = nn.Linear(hidden_dims[i], hidden_dims[i + 1])

        # Define the output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        """
        Forward pass method
        Required method by pytorch

        in: batch of input data [torch.float32 tensor]
        """
        out = self.input_layer(x)
        out = F.relu(out)
        for i in range(len(self.hidden_layers)):
            out = self.hidden_layers['h_{}'.format(i)](out)
            out = F.relu(out)
        return self.output_layer(out)
