import os
import copy
import torch
from torch import nn
from pathlib import Path
from collections import deque
import random, datetime, numpy as np, cv2 
# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation

#NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros

import os

class MarioNet(nn.Module):
  '''mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  '''
  def __init__(self, input_dim, output_dim):
      super().__init__()
      c, h, w = input_dim

      if h != 84:
          raise ValueError(f"Expecting input height: 32, got: {h}")
      if w != 84:
          raise ValueError(f"Expecting input width: 32, got: {w}")

      self.online = nn.Sequential(
          nn.Conv2d(in_channels=c, out_channels=32, kernel_size=5, stride=2, padding=1),
          nn.ReLU(),
          nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
          nn.ReLU(),
          nn.MaxPool2d(3),
          
          nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
          nn.ReLU(),
          nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
          nn.MaxPool2d(2),
          
          nn.Flatten(),
          nn.Linear(1024, 128),
          nn.ReLU(),
          nn.Linear(128, output_dim)
      )

      self.target = copy.deepcopy(self.online)

      # Q_target parameters are frozen.
      for p in self.target.parameters():
          p.requires_grad = False

  def forward(self, input, model):
      if model == 'online':
          return self.online(input)
      elif model == 'target':
          return self.target(input)

class Mario: 
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()
        
        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device='cuda')

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5   # no. of experiences between saving Mario Net
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.9
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.burnin = 1e5  # min. experiences before training
        self.learn_every = 3   # no. of experiences between updates to Q_online
        self.sync_every = 1e4   # no. of experiences between Q_target & Q_online sync
        
    def learn(self):
      if self.curr_step % self.sync_every == 0:
          self.sync_Q_target()

      if self.curr_step % self.save_every == 0:
          self.save()

      if self.curr_step < self.burnin:
          return None, None

      if self.curr_step % self.learn_every != 0:
          return None, None

      # Sample from memory
      state, next_state, action, reward, done = self.recall()

      # Get TD Estimate
      td_est = self.td_estimate(state, action)

      # Get TD Target
      td_tgt = self.td_target(reward, next_state, done)

      # Backpropagate loss through Q_online
      loss = self.update_Q_online(td_est, td_tgt)

      return (td_est.mean().item(), loss)

    def update_Q_online(self, td_estimate, td_target) :
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())
    
    def td_estimate(self, state, action):
        current_Q = self.net(state, model='online')[np.arange(0, self.batch_size), action] # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
    

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state).cuda() if self.use_cuda else torch.FloatTensor(next_state)
        action = torch.LongTensor([action]).cuda() if self.use_cuda else torch.LongTensor([action])
        reward = torch.DoubleTensor([reward]).cuda() if self.use_cuda else torch.DoubleTensor([reward])
        done = torch.BoolTensor([done]).cuda() if self.use_cuda else torch.BoolTensor([done])

        self.memory.append( (state, next_state, action, reward, done,) )

    
    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def save(self):
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")
        
    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate
    