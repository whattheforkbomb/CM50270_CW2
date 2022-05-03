import gym
from typing import Union

from nes_py.wrappers import JoypadSpace

class Config():
    """A class that stores core configuration variables."""
    def __init__(self, env: Union[gym.Env, JoypadSpace], env_name : str) -> None:
        self.env = env
        self.env_name = env_name
        self.set_env_params()

    def add(self, **kwargs) -> None:
        """Add parameters to the class."""
        for key, val in kwargs.items():
            setattr(self, key, val)

    def set_env_params(self) -> None:
        """Sets crucial environment variables."""
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.input_shape = self.observation_space.shape
        self.n_actions = self.action_space.n
