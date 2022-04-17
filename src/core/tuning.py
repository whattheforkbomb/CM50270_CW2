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
    """A class dedicated to training and tuning models."""
    def __init__(self, env: gym.Env, agent: BaseAgent, model: nn.Module, exp_buffer: ExperienceBuffer, optimizer: torch.optim, device: str, stop_reward: int = 30) -> None:
        self.env = env
        self.agent = agent
        self.model = model
        self.buffer = exp_buffer
        self.optimizer = optimizer
        self.device = device
        self.reward_threshold = stop_reward

        self.start_time = time.time()
        self.total_frames = 0
        self.total_rewards = []

    def train(self, episode_count: int, batch_size: int, entropy_beta: float, clip_grad: float) -> None:
        """Train the model on a given number of episodes."""
        done = False
        batch = []
        for i_episode in range(episode_count):
            print(f'({i_episode}/{episode_count})' , end="")
            for step_idx, exp in enumerate(self.buffer):
                batch.append(exp)

                # Stop training if reached reward threshold
                new_rewards = self.buffer.pop_total_rewards()
                if new_rewards and self.reward_tracker(new_rewards[0], step_idx):
                    save_model()
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
                temp = rewards - state_value.squeeze(-1).detach()
                log_action_probs = temp * log_probs[range(batch_size), actions]
                loss_policy = log_action_probs.mean()
                probs = F.softmax(policy, dim=1)
                entropy_loss = entropy_beta * (probs * log_probs).sum(dim=1).mean()

                # Backpropagate policy gradients
                loss_policy.backward(retain_graph=True)
            
                # Backpropagate entropy and state-value gradients
                total_loss = entropy_loss + loss_sv
                total_loss.backward()
                nn_utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                self.optimizer.step()

                # Get total loss
                total_loss += loss_policy
                print(f'state loss {loss_sv:.3f}, policy loss {loss_policy:.3f}, train loss {total_loss:.3f}')
            
            # Terminate if training complete
            if done:
                break
        self.env.close()
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