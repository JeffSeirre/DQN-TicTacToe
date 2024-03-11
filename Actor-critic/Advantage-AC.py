import numpy as np
import torch
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt


class Network(nn.module):
    def __init__(self, input_dim, output_dim):

        super.__init__()

        # Define the layers of the network
        self.Network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softplus()
        )
        
    def forward(self, x):
        return self.Network(x)


def main():
    ...

if __name__ == 'main':
    main()