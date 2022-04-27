from collections import namedtuple
from typing import Union
import numpy as np

from utils.config import Config
from utils.helper import to_tensor, to_numpy

import torch
import torch.nn as nn

Experience = namedtuple('Experience', ['states', 'actions', 'rewards', 'dones'])

class A2CAgent():
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

    def normalize_states(self, states: list) -> Union[np.array, torch.Tensor]:
        """Normalize a given list of states."""
        if not isinstance(states, torch.Tensor):
            states = np.asarray(states)
        return (1.0 / 255) * states

    def train(self, target_score: int = 300, print_every: int = 100) -> None:
        """Train the agent."""
        print(f'Running training with N-Steps: {self.config.rollout_size}')
        for i_episode in range(self.config.num_episodes):
            states, actions, rewards, dones = [], [], [], []
            state = self.env.reset()

            # Get training data
            for _ in range(self.config.rollout_size):
                state = to_tensor(state).to(self.config.device)
                
                action_probs = self.network.forward(state.unsqueeze(0))[0]
                action = np.random.choice(action_probs.size(dim=1), p=to_numpy(action_probs).ravel())
                next_state, reward, done, _ = self.env.step(action)
                
                # Add values to lists
                states.append(state.cpu().numpy())
                actions.append(action)
                rewards.append(reward)
                dones.append(done)

                # Update state
                state = next_state

                # Check if episode complete
                if done:
                    break
            
            # Add episode actions to logger
            self.config.logger.add_actions(actions)

            # Reflect on training data
            exp = Experience(to_tensor(states).to(self.config.device), to_tensor(actions), np.array(rewards), np.array(dones))
            self.reflect(exp)

            # Output episode info
            first_episode = i_episode == 0
            last_episode = i_episode+1 == self.config.num_episodes
            if i_episode % print_every == 0 or first_episode or last_episode:
                episode_actions = self.config.logger.actions[i_episode]
                episode_return = self.config.logger.avg_returns[i_episode]
                episode_loss = self.config.logger.total_losses[i_episode]
                if first_episode or last_episode:
                    print(f'({i_episode+1}', end='')
                else:
                    print(f'({i_episode}', end='')
                print(f'/{self.config.num_episodes})\tEpisode actions: {episode_actions}\tAvg return: {episode_return:.3f}\tTotal loss: {episode_loss:.3f}')

            # Save model if goal achieved
            avg_return = self.config.logger.avg_returns[i_episode]
            if avg_return >= target_score:
                print(f'Environment solved in {i_episode} episodes! Avg score: {avg_return:.3f}')
                self.save_model()
                break

    def reflect(self, experience: namedtuple) -> None:
        """Reflect on the generated experiences by training the model."""
        # Calculate ground truth state-values
        state_values_true = self.get_state_values(experience)

        # Get the action probs and state-value estimates
        states = experience.states.to(self.config.device)
        action_probs, state_values_est = self.network.forward(self.normalize_states(states))
        action_log_probs = action_probs.log()
        a = experience.actions.type(torch.LongTensor).view(-1, 1).detach()
        action = action_log_probs.gather(1, a.to(self.config.device))

        advantages = state_values_true.to(self.config.device) - state_values_est

        entropy = (action_probs * action_log_probs).sum(1).mean()
        action_gain = (action * advantages).mean()
        value_loss = advantages.pow(2).mean()
        total_loss = value_loss - action_gain - self.config.entropy_weight * entropy

        # Add total loss to logger
        self.config.logger.add_loss(total_loss)

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
        svs.append(next_return.cpu().numpy())
        dones = np.flip(experience.dones)

        avg_rewards.append(next_return.cpu().mean())
        # Iterate over each reward (ignoring last state)
        for r in range(1, len(rewards)):
            # Set the current return
            if not dones[r]:
                cur_return = rewards[r] + next_return * self.config.gamma
            else:
                cur_return = 0
        
            # Add return to list and update to next one
            svs.append(to_numpy(cur_return))
            next_return = cur_return.cpu()
            avg_rewards.append(next_return.mean())
        
        # Add avg return to logger
        self.config.logger.add_return(np.array(avg_rewards).mean())
        
        svs.reverse()
        return to_tensor(svs)

    def save_model(self) -> None:
        """Saves the model's parameters."""
        torch.save(self.network.state_dict(), f'saved_models/a2c.pt')
    
    def load_model(self) -> None:
        """Load a saved model's parameters."""
        self.network.load_state_dict(torch.load(f'saved_models/a2c.pt', map_location=self.config.device))
