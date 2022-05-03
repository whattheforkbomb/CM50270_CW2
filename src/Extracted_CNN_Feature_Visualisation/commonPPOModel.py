import imp
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
import gym

import torch
import torch.nn as nn

from stable_baselines3.common import atari_wrappers
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

INPUT_TYPE = {
    "RAW": 0,
    "DOWNSAMPLE": 1,
    "PIXEL": 2,
    "RECTANGLE": 3,
    "raw": 0,
    "down": 1,
    "pixel": 2,
    "rect": 3
}

ACTION_SPACE = {
    "sm" : SIMPLE_MOVEMENT, 
    "ro" : RIGHT_ONLY, 
    "cm" : COMPLEX_MOVEMENT,
    "simple" : SIMPLE_MOVEMENT, 
    "right" : RIGHT_ONLY, 
    "complex" : COMPLEX_MOVEMENT,
}
    
def create_vec_env(version, action_space, random, n, monitor = False):
    def create_env():
        env_name = "SuperMarioBros" if not random else "SuperMarioBrosRandomStage"
        env = gym_super_mario_bros.make('{}-v{}'.format(env_name, INPUT_TYPE[version]))
        env = JoypadSpace(env, ACTION_SPACE[action_space])
        if monitor:
            from stable_baselines3.common.monitor import Monitor
            env = Monitor(env, allow_early_resets=True) # required to log rollout stuff (e.g. reward)
        # Convert to greyscale and downsample to 84x84
        return atari_wrappers.AtariWrapper(env, noop_max=5, terminal_on_life_loss=False)

    return (DummyVecEnv([lambda: create_env() for _ in range(n)]), (INPUT_TYPE[version], action_space, int(random), n))

# Derived from: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))