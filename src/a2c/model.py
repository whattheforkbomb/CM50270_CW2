import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class ACNetwork(nn.Module):
    """
    A neural network architecture for the actor-critic method to solve the Super Mario Bros environment.

    Parameters:
        input_shape (int) - image input shape
        n_actions (int) - number of actions
        hidden_sizes (list) - list of integers for hidden layer sizes
    """
    def __init__(self, input_shape: tuple[int], n_actions: int) -> None:
        super().__init__()
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_size(input_shape)

        # Dense layers
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        ) # policy

        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        ) # action-value function

    def _get_conv_size(self, input_shape: tuple[int]) -> int:
        """Gets the convolutional layers output size."""
        out = self.conv(torch.zeros(1, *input_shape))
        return int(np.prod(out.size()))

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagates the features through the network."""
        x = x.float() / 256
        conv_out = self.conv(x).view(x.size()[0], -1) # flatten

        # 1. Return policy with probability distribution over actions
        # 2. Return single approximation of state value
        return F.softmax(self.actor(conv_out), dim=1), self.critic(conv_out)