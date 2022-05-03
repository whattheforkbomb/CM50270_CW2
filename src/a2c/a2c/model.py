import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ACNetwork(nn.Module):
    """
    A basic actor-critic neural network.

    Parameters:
        input_shape (tuple[int]) - image input dimensions
        n_actions (int) - number of actions
    """
    def __init__(self, input_shape: tuple[int], n_actions: int) -> None:
        super().__init__()
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.MaxPool2d(2),
        )
        conv_out_size = self._get_conv_size(input_shape)

        # Dense layers
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        ) # policy

        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ) # action-value function

    def _get_conv_size(self, input_shape: tuple[int]) -> int:
        """Gets the convolutional layers output size."""
        out = self.conv(torch.zeros(1, *input_shape))
        return int(np.prod(out.size()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagates the features through the network."""
        conv_out = self.conv(x).view(x.size()[0], -1) # flatten

        # 1. Return policy with probability distribution over actions
        # 2. Return single approximation of state value
        return F.softmax(self.actor(conv_out), dim=1), self.critic(conv_out)
