import os
import math
import gym
from collections import Counter
from IPython import display
import pandas as pd
from typing import Union

from a2c.agent import A2CAgent
from utils.logger import Logger
from utils.helper import to_tensor, normalize_states, human_format_number

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

class Plotter():
    """Manages visualisation of logger values and rendering of the environment."""
    def __init__(self, logger: Logger, rollout_size: int) -> None:
        self.logger = logger
        self.rollout_size = rollout_size

        sns.set_style("whitegrid")
        self.colours = sns.color_palette("muted")
        self.ticksize = 12

        os.makedirs('plots', exist_ok=True)

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

    def actions_chart(self, data: list, subplot_title: str, colour_idx: int = 0, figsize: tuple[float, float] = (10, 12), **kwargs) -> None:
        """Creates a bar chart with the count for each action taken in the environment across all episodes."""
        os.makedirs('plots/actions', exist_ok=True)

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

        filename = 'actions-' + '-'.join(subplot_title.split())
        plt.suptitle(f'Actions taken in the environment ({int(data_size)}{letter} episodes)'.title(), fontsize=16, fontweight='bold')
        plt.savefig(f'plots/actions/{filename}.png')
        plt.show()

    @staticmethod
    def num_to_list(data: list) -> list[list]:
        """Fixes a bug in the data if there are numbers in a list of lists."""
        correct_data = []
        for idx, item in enumerate(data):
            if isinstance(item, (int, float)):
                temp = data[idx-1].copy()
                temp.append(item)
                correct_data.append(temp)

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

    def plot_episodic_data(self, data: list, data_name: str, subplot_title: str, colour_idx: int = 0, figsize: tuple[float, float] = (10, 12), **kwargs) -> None:
        """Creates a plot displaying the specified data across each episode."""
        folder_name = '_'.join(data_name.lower().split())
        os.makedirs(f'plots/{folder_name}', exist_ok=True)

        data_size, letter = human_format_number(math.floor(len(data)))

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(range(len(data)), data, color=self.colours[colour_idx], **kwargs)

        plt.xticks(fontsize=self.ticksize)
        plt.yticks(fontsize=self.ticksize)
        plt.xlabel("...", labelpad=12)
        ax.set_xlabel('Episodes', fontweight='bold', fontsize=self.ticksize+2)
        ax.set_ylabel(f'{data_name}', fontweight='bold', fontsize=self.ticksize+2)
        ax.set_title(f'{subplot_title}', pad=16)

        plt.suptitle(f'{data_name} obtained in the environment ({int(data_size)}{letter} episodes)'.title(), fontsize=16, fontweight='bold')
        data_name = '-'.join(data_name.split())
        subplot_title = '-'.join(subplot_title.split())
        filename = data_name + '-' + subplot_title
        plt.savefig(f'plots/{folder_name}/{filename.lower()}.png')
        plt.show()

    def plot_all_metrics(self, ranges: list[Union[list[str], list[int, int], list[str, int], list[int, str]]]) -> None:
        """
        Create a set of plots for each type of key in the agents logger.
        
        Parameters:
        - ranges (list) - a list of lists that contains two integers for [start, end]. Special items include: ['all'] for all episodes, and [start, ''] or ['', end] for edge cases.

        Examples:
        >> Rollouts 10, max episodes at 200k
        >> plot_all_metrics(ranges=[['all'], [110000, '']]) 
        >> # ['10 rollouts', '10 rollouts at episodes 100k to 200k'] 
        """
        sub_titles = self._set_subtitles(ranges)

        for key in self.logger.keys:
            if key not in ['env_info', 'save_batch_stats', 'log_probs', 'entropys']:
                data = getattr(self.logger, key)
                if isinstance(data[0], list):
                    data = self.num_to_list(data)

                for idx, item in enumerate(ranges):
                    if key == 'actions':
                        func = lambda x, y: self.actions_chart(data=x, subplot_title=sub_titles[idx], figsize=(10, 7), width=y)
                        y = 0.75
                    else:
                        func = lambda x, y: self.plot_episodic_data(data=x, data_name=y, subplot_title=sub_titles[idx], figsize=(12, 7))
                        y = ' '.join(key.split('_')).title()
                    
                    if item[0] == 'all':
                        func(data, y)
                    elif item[0] == '':
                        func(data[:item[1]], y)
                    elif item[1] == '':
                        func(data[item[0]:], y)
                    else:
                        func(data[item[0]:item[1]], y)

    def _set_subtitles(self, ranges: list[list]) -> list[str]:
        """Create the subtitles for the plots from a list of ranges."""
        titles = []
        rollout_str = f'{self.rollout_size} rollouts'
        ep_str = f'{rollout_str} at episodes'

        for item in ranges:
            if item[0] == 'all':
                item = f'{rollout_str}'
                titles.append(item)
            elif len(item) > 1 and any([isinstance(item[0], int), isinstance(item[1], int)]):
                if item[1] == '':
                    item[1] = 200000
                if item[0] == '':
                    item[0] = 0
                
                start_num, start_letter = human_format_number(item[0])
                end_num, end_letter = human_format_number(item[1])

                titles.append(f'{ep_str} {int(start_num)}{start_letter} to {int(end_num)}{end_letter}')
            else:
                raise ValueError
        
        return titles

    def data_to_csv(self) -> None:
        """Converts a provided set of data into a CSV file."""
        directory = 'stats_csvs'
        folder_name = f'{directory}/r{self.rollout_size}'
        os.makedirs(folder_name, exist_ok=True)
        
        for key in self.logger.keys:
            if key not in ['env_info', 'save_batch_stats']:
                data = getattr(self.logger, key)
                if isinstance(data[0], list):
                    header = [f'r{x}' for x in range(1, self.rollout_size+1)]
                    header.append('next_state')
                    data = pd.DataFrame(self.num_to_list(data), columns=header)
                else:
                    header = [key]
                    data = pd.DataFrame(data, columns=header)
            
                data.to_csv(f'{folder_name}/{key}.csv', index=False)
