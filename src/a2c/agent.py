import numpy as np

from utils.storage import Storage
from utils.config import Config
from utils.logger import Logger
from utils.helper import normalize_states, set_device, to_tensor

import torch
import torch.nn as nn

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
        self.network = config.network
        self.optimizer = config.optimizer
        self.logger = Logger()

    def train(self, target_score: int = 300, save_count: int = 1000, print_every: int = 100) -> None:
        """Train the agent."""
        print(f'Running training with rollout length {self.config.rollout_size}.')
        for i_episode in range(self.config.num_episodes):
            actions, log_probs, entropys, env_info = [], [], [], []
            state = self.env.reset()
            self.storage = Storage(self.config.rollout_size) # reset each episode

            # Get training data
            for _ in range(self.config.rollout_size):
                # Calculate policy
                state = to_tensor(state).to(self.config.device)
                action_probs, state_value = self.network.forward(normalize_states(state.unsqueeze(0)))
                
                # Get predictions
                preds = self.get_predictions(action_probs)
                
                # Step through the enviornment
                next_state, reward, done, info = self.env.step(preds['action'].item())

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
            self._output_stats(i_episode, print_every)

            # Save model every save count or when goal has been achieved
            if self._save_model_conditions(i_episode, save_count, target_score):
                break

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
        next_state = to_tensor(next_state).to(self.config.device)
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

        # Add info to logger
        self.logger.add(
            avg_advantages=entries.advantages.mean().item(), 
            avg_returns=entries.next_returns.mean().item(), 
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

    def _output_stats(self, i_episode: int, print_every: int) -> None:
        """Returns information to the console."""
        first_episode = i_episode == 0
        last_episode = i_episode+1 == self.config.num_episodes

        if i_episode % print_every == 0 or first_episode or last_episode:
            episode_actions = self.logger.actions[i_episode]
            episode_return = self.logger.avg_returns[i_episode]
            episode_loss = self.logger.total_losses[i_episode]

            if first_episode or last_episode:
                print(f'({i_episode+1}', end='')
            else:
                print(f'({i_episode}', end='')

            print(f'/{self.config.num_episodes}) ', end='')
            if not isinstance(episode_actions, int) and len(episode_actions) <= 10:
                print(f'Episode actions: {episode_actions}', end='')
            print(f'\tAvg return: {episode_return:.3f}\tTotal loss: {episode_loss:.3f}')

    def _save_model_conditions(self, i_episode: int, save_count: int, target_score: int) -> bool:
        avg_return = self.logger.avg_returns[i_episode]
        total_loss = self.logger.total_losses[i_episode]
        if i_episode+1 % save_count == 0:
            print(f'Saved model at episode {i_episode+1}. Avg return: {avg_return:.3f}\tTotal loss: {total_loss:.3f}')
            
        if avg_return >= target_score:
            print(f"Completed environment in {i_episode+1}'s! Avg return: {avg_return:.3f}\tTotal loss: {total_loss:.3f}")
            self.save_model()
            return True
        return False

    def save_model(self) -> None:
        """Saves the model's parameters."""
        param_dict = dict(
            model=self.network.state_dict(), 
            config=self.config,
            logger=self.logger
        )
        torch.save(param_dict, f'saved_models/a2c.pt')
    
    def load_model(self) -> None:
        """Load a saved model's parameters."""
        checkpoint = torch.load(f'saved_models/a2c.pt', map_location=set_device())
        self.network.load_state_dict(checkpoint.get('model'))
        self.config = checkpoint.get('config')
        self.logger = checkpoint.get('logger')
        print('Loaded A2C model.')
