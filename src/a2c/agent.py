from typing import Union
import numpy as np

from core.experience import ExperienceBuffer
from utils.config import Config
from utils.helper import to_numpy, to_tensor

import torch
import torch.nn as nn

class BaseAgent():
    """Base class for RL agents."""
    def save_model(self, filename: str, params_dict: dict) -> None:
        """
        Saves a models as the given filename.

        Parameters:
        - params_dict (dict) - parameters to store
        - filename (str) - name of the saved model
        """
        torch.save(params_dict, f'saved_models/{filename}.pt')
    
    def load_model(self, model: nn.Module, filename: str) -> None:
        """
        Load a saved model based on the given filename and stores its parameters to a given model.

        Parameters:
        - model (nn.Module) - model to store parameters to
        - filename (str) - name of the model to load    
        """
        checkpoint = torch.load(f'saved_models/{filename}.pt')

        # Store variables as model attributes
        for key, val in checkpoint.items():
            if key != 'parameters':
                setattr(model, key, val)

        # Load model parameters
        model.load_state_dict(checkpoint['parameters'])

        print("Model loaded. Utility variables available:")
        print("  ", end='')
        for idx, key in enumerate(checkpoint.keys()):
            if key != 'parameters':
                if len(checkpoint.keys())-1 == idx:
                    print(f'{key}.')
                else:
                    print(f"{key}, ", end='')

class A2CAgent(BaseAgent):
    """
    A basic representation of an Advantage Actor-Critic agent.
    
    Parameters:
    - config (Config) - class of config variables
    """
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.total_steps = 0
        self.states = self.task.reset()
    
    def train(self):
        """Train the agent."""
        config = self.config
        buffer = ExperienceBuffer(config.rollout_size)
        states = self.states

        # Iterate over episodes
        for i_episode in range(config.num_episodes):
            if i_episode == 0 and i_episode % 10 == 0:
                print(f'({i_episode+1}/{config.num_episodes}) ', end="")
            
            # Iterate over buffer
            for _ in range(config.rollout_size):
                states = self.normalize_states(states)
                preds = self.network(states)
                next_states, rewards, dones, info = self.task.step(to_numpy(preds))
                ################## HERE ##################
                self.record_online_return(info) # <------
                ################## HERE ##################
                rewards = np.sign(rewards) # {-1, 1}
                buffer.add(preds)
                buffer.add({
                    'reward': to_tensor(rewards).to(config.device).unsqueeze(-1),
                    'mask': to_tensor(1 - dones).to(config.device).unsqueeze(-1)
                })

                # Update states and steps
                states = next_states
                self.total_steps += config.num_agents
            
            self.states = states
            preds = self.network(self.normalize_states(states))
            buffer.add(preds)
            buffer.initalise_values()

            # Initialise advantages and returns
            advantages = to_tensor(np.zeros(config.num_agents, 1)).to(config.device)
            returns = preds['v'].detach()

            # Iterate over each item in the buffer
            for i in reversed(range(config.rollout_size)):
                # Calculate returns, td error and advantages
                returns *= buffer.reward[i] + config.gamma * buffer.mask[i]
                td_error = buffer.reward[i] + config.gamma * buffer.mask[i] * buffer.v[i + 1] - buffer.v[i]
                advantages *= config.gamma * buffer.mask[i] + td_error

                # Update buffer
                buffer.advantage[i] = advantages.detach()
                buffer.returns[i] = returns.detach()

            # Get data from buffer
            samples = buffer.sample(['log_pi_a', 'v', 'returns', 'advantage', 'entropy'])

            # Calculate losses
            loss_policy = -(samples.log_pi_a * samples.advantage).mean()
            loss_value = 0.5 * (samples.returns - samples.v).pow(2).mean()
            loss_entropy = samples.entropy.mean()

            # Perform a training step
            self.optimizer.zero_grad()
            loss_total = loss_policy - config.entropy_weight * loss_entropy + config.value_loss_weight * loss_value
            loss_total.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), config.grad_clip)
            self.optimizer.step()

    def normalize_states(self, states: list) -> Union[np.array, torch.Tensor]:
        """Normalize a given list of states."""
        if not isinstance(states, torch.Tensor):
            states = np.asarray(states)
        return (1.0 / 255) * states

