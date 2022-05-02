import os
import numpy as np

from utils.storage import Storage
from utils.config import Config
from utils.logger import Logger
from utils.helper import human_format_number, normalize_states, to_tensor

import torch
import torch.nn as nn

class A2CAgent():
    """
    A basic representation of an Advantage Actor-Critic agent.
    
    Parameters:
    - config (Config) - class of config variables
    - device (str) - name of primary CUDA device (cpu, cuda:0, ...)
    """
    def __init__(self, config: Config, device: str) -> None:
        self.config = config
        self.network = self.config.network
        self.optimizer = config.optimizer
        self.logger = Logger()
        
        # Add device to config
        self.config.add(device = {
            'primary': device,
            'trained_count': torch.cuda.device_count()
        })

        # Parallelise if multiple devices
        if torch.cuda.device_count() > 1:
            device_names = [torch.device(f'cuda:{i}') for i in range(1, torch.cuda.device_count()-1)]
            self.network = nn.DataParallel(self.network, device_ids=device_names, output_device=device)
        
        # Set output to primary device
        self.network = self.network.to(device)

    def train(self, print_every: int = 100, save_count: int = 1000) -> None:
        """Train the agent."""
        print(f'Running training with rollout length {self.config.rollout_size} on {self.config.num_episodes} episodes.')
        # Set saving batch metrics
        self.start_ep = 1
        self.ep_batch_score = 0.
        self.ep_batch_losses = []

        # Iterate over each episode
        for i_episode in range(self.config.num_episodes):
            actions, log_probs, entropys, env_info = [], [], [], []
            state = self.config.env.reset()
            self.storage = Storage(self.config.rollout_size) # reset each episode

            # Get training data
            for _ in range(self.config.rollout_size):
                # Calculate policy
                state = to_tensor(state).to(self.config.device['primary'])
                action_probs, state_value = self.network.forward(normalize_states(state.unsqueeze(0)))
                
                # Get predictions
                preds = self.get_predictions(action_probs)
                
                # Step through the enviornment
                next_state, reward, done, info = self.config.env.step(preds['action'].item())

                # Add info to storage
                self.storage.add(
                    states=state,
                    actions=preds['action'],
                    rewards=to_tensor(np.sign(reward)).unsqueeze(-1),
                    next_states=next_state,
                    dones=to_tensor(1 - done).unsqueeze(-1),
                    returns=state_value,
                    log_probs=preds['log_prob'],
                    entropys=preds['entropy']
                )

                # Add values to lists for logger
                actions.append(preds['action'].item())
                log_probs.append(preds['log_prob'].item())
                entropys.append(preds['entropy'].item())
                env_info.append(info)

                # Update state
                state = next_state

                # Check if episode complete
                if done:
                    break

            # Add episode info to logger
            self.logger.add(
                actions=actions,
                log_probs=log_probs,
                entropys=entropys,
                env_info=info
            )

            # Update network
            self._update_network(state)

            # Output episode info
            self._output_stats(i_episode, print_every, save_count)

    @staticmethod
    def get_predictions(action_probs: torch.Tensor) -> dict:
        """Get the network prediction calculations."""
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return {
            'action': action.cpu(),
            'log_prob': log_prob.cpu(),
            'entropy': entropy.cpu()
        }

    def _update_network(self, next_state: torch.Tensor) -> None:
        """Update the networks based on the generated experience."""
        # Get the action probs and state-value estimates
        next_state = to_tensor(next_state).to(self.config.device['primary'])
        action_probs, next_return = self.network.forward(normalize_states(next_state).unsqueeze(0))

        # Get predictions
        next_preds = self.get_predictions(action_probs)

        # Add info to storage
        self.storage.add(
            actions=next_preds['action'],
            returns=next_return,
            log_probs=next_preds['log_prob'],
            entropys=next_preds['entropy']
        )

        # Add episode info to logger
        self.logger.add(
            actions=next_preds['action'].item(),
            log_probs=next_preds['log_prob'].item(),
            entropys=next_preds['entropy'].item(),
        )

        # Set empty key placeholder values
        self.storage.placeholder()

        advantage = torch.tensor(np.zeros((1, 1)))
        next_return = next_return.cpu().detach()
        # Backwards iteration over each step
        for i in reversed(range(self.config.rollout_size)):
            # Increment advantage and next return
            next_return = self.storage.rewards[i] + self.config.discount * self.storage.dones[i] * next_return
            advantage = next_return - self.storage.returns[i].cpu().detach()

            # Add data to storage backwards
            self.storage.advantages[i] = advantage.detach()
            self.storage.next_returns[i] = next_return.detach()

        # Retrieve values from storage
        entries = self.storage.retrieve(['log_probs', 'returns', 'next_returns', 'advantages', 'entropys'])

        # Calculate losses
        policy_loss = -(entries.log_probs * entries.advantages).mean()
        value_loss = 0.5 * (entries.next_returns - entries.returns.cpu()).pow(2).mean()
        entropy_loss = entries.entropys.mean()
        total_loss = policy_loss - self.config.entropy_weight * entropy_loss + self.config.value_loss_weight * value_loss

        # Update episode batch metrics
        avg_return = entries.next_returns.mean().item()
        self.ep_batch_score += avg_return
        self.ep_batch_losses.append(total_loss)

        # Add info to logger
        self.logger.add(
            avg_advantages=entries.advantages.mean().item(), 
            avg_returns=avg_return, 
            total_losses=total_loss.item(),
            policy_losses=policy_loss.item(),
            value_losses=value_loss.item(),
            entropy_losses=entropy_loss.item()
        )

        # Backpropagate the network
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.grad_clip)
        self.optimizer.step()

    def _output_stats(self, i_episode: int, print_every: int, save_count: int) -> None:
        """Returns information to the console."""
        first_episode = i_episode == 0
        last_episode = i_episode+1 == self.config.num_episodes

        if first_episode or last_episode or i_episode % print_every == 0:
            episode_return = self.logger.avg_returns[i_episode]
            episode_loss = self.logger.total_losses[i_episode]

            if first_episode or last_episode:
                i_episode += 1

            ep_idx, ep_letter = human_format_number(i_episode)
            ep_total_idx, ep_total_letter = human_format_number(self.config.num_episodes)

            if ep_letter == '':
                ep_idx = int(ep_idx)
                
            print(f'({ep_idx}{ep_letter}/{int(ep_total_idx)}{ep_total_letter}) ', end='')
            print(f'Episode avg return: {episode_return:.3f}\tEpisode total loss: {episode_loss:.3f}\tAccumulated batch total avg return: {self.ep_batch_score:.3f}')

            # Save model every 'print_every' steps
            self._save_model_conditions(i_episode, save_count, [int(ep_idx), ep_letter])

    def _save_model_conditions(self, i_episode: int, save_count: int, ep_items: list) -> None:
        if i_episode % save_count == 0:
            # Calculate avg batch loss
            self.ep_batch_loss = torch.stack(self.ep_batch_losses).mean().item()

            # Add values to logger
            self.logger.add(save_batch_stats={
                'ep_range': [self.start_ep, i_episode], 
                'avg_return': self.ep_batch_score,
                'avg_total_loss': self.ep_batch_loss
            })

            # Save model
            filename = f'a2c_rollout{self.config.rollout_size}_ep{ep_items[0]}{ep_items[1]}'.lower()
            self.save_model(filename)
            print(f"Saved model at episode {i_episode} as: '{filename}'.")
            print(f"  Batch total avg return: {self.ep_batch_score:.3f}")
            print(f"  Batch avg total loss: {self.ep_batch_loss:.3f}")

            # Reset batch values
            self.start_ep = i_episode+1
            self.ep_batch_score = 0.
            self.ep_batch_losses.clear()

    def save_model(self, filename: str, folder_name: str = 'saved_models') -> None:
        """Saves a model's state dict, config object and logger object to the desired 'folder_name'."""
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        param_dict = dict(
            model=self.network.state_dict(), 
            config=self.config,
            logger=self.logger
        )
        torch.save(param_dict, f'{folder_name}/{filename}.pt')
    
    def load_model(self, filename: str, device: str, folder_name: str = 'saved_models') -> None:
        """Load a saved model's parameters. Must be stored within the desired 'folder_name'."""
        checkpoint = torch.load(f'{folder_name}/{filename}.pt', map_location=device)

        self.config = checkpoint.get('config')
        self.logger = checkpoint.get('logger')

        self.config.device = device
        self.network = self.config.network.to(device)
        self.optimizer = self.config.optimizer
        self.network.load_state_dict(checkpoint.get('model'))
        print(f"Loaded A2C model: '{filename}'.")
