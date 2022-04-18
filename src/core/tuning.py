import time
import gym
import numpy as np

from a2c.agent import BaseAgent
from core.experience import ExperienceBuffer
from core.utils import save_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils

class Tuning():
    """
    A class dedicated to training and tuning models.
    
    Parameters:
    - agent (BaseAgent) - the agent to use for unpacking batches
    - model (nn.Module) - the model used to train
    - exp_buffer (ExperienceBuffer) - experience buffer for getting data samples
    - optimizer (torch.optim) - optimizer used for training the model
    - stop_reward (int) - mean reward obtained to stop training
    """
    def __init__(self, agent: BaseAgent, model: nn.Module, exp_buffer: ExperienceBuffer, optimizer: torch.optim, stop_reward: int = 30) -> None:
        self.agent = agent
        self.model = model
        self.buffer = exp_buffer
        self.optimizer = optimizer
        self.reward_threshold = stop_reward

        self.total_frames = 0
        self.total_rewards = []

    def train(self, episode_count: int, batch_size: int, entropy_beta: float, clip_grad: float, filename: str) -> None:
        """
        Train the model on a given number of episodes.
        
        Parameters:
        - episode_count (int) - number of episodes to train the model on
        - batch_size (int) - size of action probabilities
        - entropy_beta (float) - metric for entropy loss
        - clip_grad (float) - metric for preventing large gradients
        - filename (str) - filename for saving the model
        """
        done = False
        batch = []
        losses_policy, losses_sv, losses_entropy, losses_total = [], [], [], []
        advantages, state_values = [], []
        for i_episode in range(episode_count):
            print(f'({i_episode}/{episode_count})' , end="")
            for step_idx, exp in enumerate(self.buffer):
                batch.append(exp)

                # Stop training if reached reward threshold
                new_rewards = self.buffer.pop_total_rewards()
                if self.reward_tracker(new_rewards[0], step_idx):
                    params_dict = {
                        'parameters': self.model.parameters(),
                        'advantages': advantages,
                        'state_values': state_values,
                        'rewards': self.total_rewards,
                        'losses_policy': losses_policy,
                        'losses_sv': losses_sv,
                        'losses_entropy': losses_entropy,
                        'losses_total': losses_total,
                        'policy': policy
                    }
                    save_model(model=self.model, filename=filename, params_dict=params_dict)
                    done = True
                    break

                # Get batch of experiences
                states, actions, rewards = self.agent.unpack_batch(batch)
                batch.clear()

                # Perform training
                self.optimizer.zero_grad()
                policy, state_value = self.model.forward(states)
                loss_sv = F.mse_loss(state_value.squeeze(-1), rewards)
                
                # Convert to normalised form
                log_probs = F.log_softmax(policy, dim=1)
                action_advantage = rewards - state_value.squeeze(-1).detach()
                log_action_probs = action_advantage * log_probs[range(batch_size), actions]
                loss_policy = log_action_probs.mean()
                probs = F.softmax(policy, dim=1)
                loss_entropy = entropy_beta * (probs * log_probs).sum(dim=1).mean()

                # Backpropagate policy gradients
                loss_policy.backward(retain_graph=True)
            
                # Backpropagate entropy and state-value gradients
                loss_total = loss_entropy + loss_sv
                loss_total.backward()
                nn_utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                self.optimizer.step()

                # Get total loss
                loss_total += loss_policy
                print(f'state loss {loss_sv:.3f}, policy loss {loss_policy:.3f}, train loss {loss_total:.3f}')

                # Append items to lists
                advantages.append(action_advantage)
                state_values.append(state_value)
                losses_policy.append(loss_policy)
                losses_sv.append(loss_sv)
                losses_entropy.append(loss_entropy)
                losses_total.append(loss_total)
            
            # Terminate if training complete
            if done:
                break

        print('Training complete.')

    def reward_tracker(self, reward: int, frame: int) -> bool:
        """
        Helper function used to track model rewards.
        
        Parameters:
        - reward (int) - reward obtained for the state
        - frame (int) - current state frame
        """
        self.total_rewards.append(reward)
        self.total_frames = frame
        mean_reward = np.mean(self.total_rewards[-100:])
        print(f'Frame {frame}: completed {len(self.total_rewards)} games, mean reward {mean_reward:.3f}, ', end="")

        if mean_reward > self.reward_threshold:
            print(f"Solved in {frame} frames!")
            return True
        return False        

    def validate(self):
        """Test the trained model."""
        pass