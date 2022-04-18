import numpy as np

from a2c.model import ACNetwork

import torch
import torch.nn.functional as F

class BaseAgent():
    """Abstract class for RL agents."""
    def inital_state(self):
        """Returns the initial (first) state of the agent."""
        return None
    
    def __call__(self, states, agent_states):
        """
        Converts observations and states into actions to take.

        Parameters:
        - states (list) - list of environment states to process
        - agent_states (list) - list of states with the same length as the previous

        Returns:
        A tuple of state-action pairs (actions, states)
        """
        raise NotImplementedError
    
    def unpack_batch(self, batch):
        """Converts a batch of samples into torch tensors."""
        raise NotImplementedError

class A2C(BaseAgent):
    """
    A basic representation of an Advantage Actor-Critic agent.
    
    Parameters:
    - gamma (float) - discount factor
    - reward_steps (int) - action approximation metric
    - net (ACNetwork) - neural network (model)
    - device (str) - device type for cuda
    """
    def __init__(self, gamma: float, reward_steps: int, net: ACNetwork, device: str) -> None:
        self.gamma = gamma
        self.reward_steps = reward_steps
        self.net = net
        self.device = device

    @torch.no_grad()
    def __call__(self, states: list, agent_states: list = None) -> tuple:
        """
        Calls the agent to obtain an array of actions from a given list of states.

        Parameters:
        - States (list) - list of states
        - agent_states (list) - an optional list of agent states

        Returns:
        Actions (np.array) and new agent states (list).
        """
        if agent_states is None:
            agent_states = [None] * len(states)
        
        if torch.is_tensor(states):
            states = states.to(self.device)
        
        probs = self.net(states)
        probs = F.softmax(probs, dim=1).data.cpu().numpy()

        actions = self.__get_actions(probs)
        return actions, agent_states
    
    def __get_actions(probs: torch.Tensor) -> list:
        """Converts action probabilities into actions by sampling them."""
        actions = []
        for prob in probs:
            actions.append(np.random.choice(len(prob), p=prob))
        return np.array(actions)

    def unpack_batch(self, batch: list) -> tuple[torch.Tensor]:
        """Converts a batch of samples into torch tensors. Returns state variables, action values, and reference values (rewards)."""
        states, actions = [], []
        rewards, last_states = [], []
        incomplete_idx = []

        # Divide batch data into lists
        for idx, exp in enumerate(batch):
            states.append(np.array(exp.state, copy=False))
            actions.append(int(exp.action))
            rewards.append(exp.reward)

            if exp.last_state is not None:
                incomplete_idx.append(idx)
                last_states.append(np.array(exp.last_state, copy=False))
            
        # Convert to tensors
        states = torch.FloatTensor(np.array(states, copy=False).to(self.device))
        actions = torch.LongTensor(actions).to(self.device)

        # Handle rewards
        rewards = np.array(rewards, dtype=np.float32)
        if incomplete_idx:
            last_states = torch.FloatTensor(np.array(last_states, copy=False)).to(self.device)
            last_vals = self.net(last_states)[1]
            last_vals = last_vals.data.cpu().numpy()[:, 0]
            last_vals *= self.gamma ** self.reward_steps
            rewards[incomplete_idx] += last_vals
        
        rewards = torch.FloatTensor(rewards).to(self.device)
        return states, actions, rewards
