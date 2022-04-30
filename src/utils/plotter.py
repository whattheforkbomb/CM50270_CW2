import gym
from IPython import display

from a2c.agent import A2CAgent
from utils.logger import Logger
from utils.helper import to_tensor, normalize_states

import torch
import matplotlib.pyplot as plt

class Plotter():
    """Manages visualisation of logger values and rendering of the environment."""
    def __init__(self, logger: Logger) -> None:
        self.logger = logger

    def video_render(self, env: gym.Env, agent: A2CAgent, steps: int = 3000) -> None:
        """Watch a video representation of an agent in a given environment."""
        state = env.reset()

        for _ in range(steps):
            state = normalize_states(to_tensor(state)).to(agent.config.device)
            action_probs = agent.config.network.forward(state.unsqueeze(0))[0]
            action = torch.distributions.Categorical(action_probs).sample().item()
            next_state, reward, done, _ = env.step(action)
            env.render()
            
            state = next_state
            
            if done:
                state = env.reset()
    
        env.close()
    
    def plot_render(self, env: gym.Env, agent: A2CAgent, steps: int = 3000) -> None:
        """Watch a plot representation of an agent in a given environment."""
        state = env.reset()
        img = plt.imshow(env.render(mode='rgb_array')) # only call this once

        for _ in range(steps):
            img.set_data(env.render(mode='rgb_array')) # just update the data
            display.display(plt.gcf())
            display.clear_output(wait=True)
            
            state = normalize_states(to_tensor(state)).to(agent.config.device)
            action_probs = agent.config.network.forward(state.unsqueeze(0))[0]
            action = torch.distributions.Categorical(action_probs).sample().item()
            next_state, reward, done, _ = env.step(action)
            
            state = next_state
            
            if done:
                state = env.reset()
        
        env.close()
