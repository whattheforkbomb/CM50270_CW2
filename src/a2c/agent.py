from collections import namedtuple
from typing import Union
import numpy as np

from utils.config import Config
from utils.helper import to_tensor, to_numpy

import torch
import torch.nn as nn

Experience = namedtuple('Experience', ['states', 'actions', 'rewards', 'dones'])

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
        self.env = config.env
        self.network = config.network_fn()
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.filename = config.filename

    def normalize_states(self, states: list) -> Union[np.array, torch.Tensor]:
        """Normalize a given list of states."""
        if not isinstance(states, torch.Tensor):
            states = np.asarray(states)
        return (1.0 / 255) * states

    def train(self) -> None:
        """Train the agent."""
        print(f'Running training with N-Steps: {self.config.rollout_size}')
        for i_episode in range(self.config.num_episodes):
            print(f'({i_episode+1}/{self.config.num_episodes}) ', end='')
            states, actions, rewards, dones = [], [], [], []
            state = self.env.reset()

            # Get training data
            for _ in range(self.config.rollout_size):
                state = to_tensor(state).to(self.config.device)
                
                action_probs = self.network.forward(state.unsqueeze(0))[0]
                action = np.random.choice(action_probs.size(dim=1), p=to_numpy(action_probs).ravel())
                next_state, reward, done, _ = self.env.step(action)
                
                # Add values to lists
                states.append(state.numpy())
                actions.append(action)
                rewards.append(reward)
                dones.append(done)

                # Update state
                state = next_state

                # Check if episode complete
                if done:
                    break
            
            # Reflect on training data
            print(f'Actions: {actions}', end='')
            exp = Experience(to_tensor(states), to_tensor(actions), np.array(rewards), np.array(dones))
            self.reflect(exp)

    def reflect(self, experience: namedtuple) -> None:
        """Reflect on the generated experiences by training the model."""
        # Calculate ground truth state-values
        state_values_true = self.get_state_values(experience)

        # Get the action probs and state-value estimates
        states = experience.states.to(self.config.device)
        action_probs, state_values_est = self.network.forward(self.normalize_states(states))
        action_log_probs = action_probs.log()
        a = experience.actions.type(torch.LongTensor).view(-1, 1).detach()
        action = action_log_probs.gather(1, a)

        advantages = state_values_true - state_values_est

        entropy = (action_probs * action_log_probs).sum(1).mean()
        action_gain = (action * advantages).mean()
        value_loss = advantages.pow(2).mean()
        total_loss = value_loss - action_gain - self.config.entropy_weight * entropy
        print(f', Total Loss: {total_loss}')

        # Backpropagate the network
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.grad_clip)
        self.optimizer.step()

    def get_state_values(self, experience: namedtuple) -> torch.Tensor:
        """Calculates the true state-values and returns them as a tensor."""
        svs = []
        rewards = np.flip(experience.rewards)
        avg_rewards = []

        # If terminal state, reward = 0
        if experience.dones[-1]:
            next_return = 0
        # Otherwise, set to value-state
        else:
            next_return = self.network.forward(self.normalize_states(experience.states))[1].detach()
        
        # Add last state
        svs.append(np.array(next_return))
        dones = np.flip(experience.dones)

        avg_rewards.append(next_return.mean())
        # Iterate over each reward (ignoring last state)
        for r in range(1, len(rewards)):
            # Set the current return
            if not dones[r]:
                cur_return = rewards[r] + next_return * self.config.gamma
            else:
                cur_return = 0
        
            # Add return to list and update to next one
            svs.append(np.array(cur_return.detach()))
            next_return = cur_return
            avg_rewards.append(next_return.mean())
        print(f', Avg return: {np.array(avg_rewards).mean():.3f}', end='')
        
        svs.reverse()
        return to_tensor(svs)
