import os
import math
import gym
from collections import Counter
from IPython import display

from a2c.agent import A2CAgent
from utils.logger import Logger
from utils.helper import to_tensor, normalize_states, human_format_number

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

class Plotter():
    """Manages visualisation of logger values and rendering of the environment."""
    def __init__(self, logger: Logger) -> None:
        self.logger = logger

        sns.set_style("whitegrid")
        self.colours = sns.color_palette("muted")
        self.ticksize = 12

        if not os.path.exists('plots'):
            os.mkdir('plots')

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

    def actions_chart(self, data: list, subplot_title: str = '', colour_idx: int = 0, figsize: tuple[float, float] = (10, 12), **kwargs) -> None:
        """Creates a bar chart with the count for each action taken in the environment across all episodes."""
        data = self._ints_to_list(data)
        actions = self._string_actions(SIMPLE_MOVEMENT)
        counts = Counter(x for x_seq in data for x in set(x_seq)) 
        data_size, letter = human_format_number(math.floor(len(data)))

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(actions, counts.values(), color=self.colours[colour_idx], **kwargs)

        plt.xticks(fontsize=self.ticksize)
        plt.yticks(fontsize=self.ticksize)
        plt.xlabel("...", labelpad=12)
        ax.set_ylabel('Counts', fontweight='bold', fontsize=self.ticksize+2)
        ax.set_xlabel('Actions', fontweight='bold', fontsize=self.ticksize+2)
        ax.set_title(f'{subplot_title}', pad=16)
        plt.suptitle(f'Actions taken in the environment ({int(data_size)}{letter.lower()} episodes)', fontsize=16, fontweight='bold')
        plt.savefig(f'plots/{subplot_title}.png')
        plt.show()

    @staticmethod
    def _ints_to_list(data: list) -> list[list]:
        """Fixes a bug in the data if there are integers in a list of lists."""
        correct_data = []
        for idx, item in enumerate(data):
            if isinstance(item, int):
                data[idx-1].append(item)
                correct_data.append(data[idx-1])
        return correct_data

    @staticmethod
    def _string_actions(actions: list) -> list[str]:
        """Returns a string representation of each action."""
        actions = []
        for item in SIMPLE_MOVEMENT:
            if len(item) == 1:
                actions.append(item[0])
            else:
                actions.append(' + '.join([x for x in item]))
        return actions

    def rewards_plot(self, data: list, subplot_title: str = '', colour_idx: int = 0, figsize: tuple[float, float] = (10, 12), **kwargs) -> None:
        """Creates a plot displaying the average returns across each episode."""
        data_size, letter = human_format_number(math.floor(len(data)))

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(range(len(data)), data, color=self.colours[colour_idx], **kwargs)

        plt.xticks(fontsize=self.ticksize)
        plt.yticks(fontsize=self.ticksize)
        plt.xlabel("...", labelpad=12)
        ax.set_xlabel('Episodes', fontweight='bold', fontsize=self.ticksize+2)
        ax.set_ylabel('Average Reward', fontweight='bold', fontsize=self.ticksize+2)
        ax.set_title(f'{subplot_title}', pad=16)
        plt.suptitle(f'Average reward obtained in the environment ({int(data_size)}{letter.lower()} episodes)', fontsize=16, fontweight='bold')
        plt.savefig(f'plots/{subplot_title}.png')
        plt.show()
