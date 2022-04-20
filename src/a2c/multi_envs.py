import gym
import numpy as np

from utils.vec_env import SubprocVecEnv

class MultiEnvs():
    """A basic representation for multiple agent environments."""
    def __init__(self, env_name: str, num_envs: int) -> None:
        envs = [gym.make(env_name) for i in range(num_envs)]
        self.name = env_name
        
        self.env = SubprocVecEnv(envs) # multi-processing
        print(self.env)

        self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.observation_space.shape))
        self.action_space = self.env.action_space
        self.n_actions = self.action_space.n

    def reset(self) -> np.array:
        """Resets all environments to their initial states."""
        return self.env.reset()

    def step(self, actions: np.array) -> tuple:
        """Takes a step through each environment with a single action."""
        return self.env.step(actions)
