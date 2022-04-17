import torch
from torch import Tensor
import torch.nn as nn

class ACNetwork(nn.Module):
    """
    A neural network architecture for the actor-critic method to solve the Super Mario Bros environment.

    Parameters:
        input_shape (int) - image input shape
        n_actions (int) - number of actions
        hidden_sizes (list) - list of integers for hidden layer sizes
        drop_prob (float) - dropout rate probability
    """
    def __init__(self, input_shape: tuple[int], n_actions: int, drop_prob: float = 0.3) -> None:
        super().__init__()
        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), # 3, 2
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_size(input_shape)

        # Dense layers
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(512, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(512, 1)
        )
        # self.dense_layers = nn.ModuleList([nn.Linear(conv_out_size, hidden_sizes[0])])
        # layer_sizes = zip(hidden_sizes[:-1, hidden_sizes[1:]])
        # self.dense_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        # self.out = nn.Linear(hidden_sizes[-1], n_actions)

    def _get_conv_size(self, input_shape: tuple[int]):
        """Gets the convolutional layers output size."""
        out = self.conv(torch.zeros(1, *input_shape))
        return int(torch.prod(out.size()))

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagates the features through the network."""
        fx = x.float() / 256 # 
        conv_out = self.conv(fx).view(fx.size()[0], -1)

        # 1. Return policy with probability distribution over actions
        # 2. Return single approximation of state value
        return self.policy(conv_out), self.value(conv_out)

        # Pass data through each dense layer
        # for linear in self.dense_layers:
            # x = F.tanh(linear(x))
            # x = self.dropout(x)
        
        # Handle output layer
        # x = F.log_softmax(self.out(x), dim=1)
        # return x