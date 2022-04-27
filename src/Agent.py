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

        self.init_exploration_rate = 0.7
        self.exploration_rate = 0.7
        self.exploration_rate_decay = 0.999999
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5   # no. of experiences between saving Mario Net
        
        self.burnin = 5e4  # min. experiences before training
        self.learn_every = 3   # no. of experiences between updates to Q_online
        self.sync_every = 1e4   # no. of experiences between Q_target & Q_online sync

        self.memory = deque(maxlen=10000)
        self.batch_size = 32

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.gamma = 0.95
    
    
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
            state = np.array(state)
            state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model='online')[0]
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        if( self.curr_step < self.burnin):
            self.exploration_rate = 1
        elif self.curr_step == self.burnin:
            self.exploration_rate = self.init_exploration_rate
        else:
            self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx


    def cache(self, state, y_pos, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        y_pos (float),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = np.array(state)
        next_state = np.array(next_state)
        
        state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
        y_pos = torch.FloatTensor([y_pos]).cuda() if self.use_cuda else torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state).cuda() if self.use_cuda else torch.FloatTensor(next_state)
        action = torch.LongTensor([action]).cuda() if self.use_cuda else torch.LongTensor([action])
        reward = torch.DoubleTensor([reward]).cuda() if self.use_cuda else torch.DoubleTensor([reward])
        done = torch.BoolTensor([done]).cuda() if self.use_cuda else torch.BoolTensor([done])

        self.memory.append( (state, y_pos, next_state, action, reward, done,) )

  
    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, y_pos, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, y_pos.squeeze(), next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_and_aux_estimate(self, state, action):
        outs = self.net(state, model='online')
        current_Q = outs[0][np.arange(0, self.batch_size), action] # Q_online(s,a)
        y_pos_est = outs[1]
        return current_Q, y_pos_est

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model='online')[0]
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model='target')[0][np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target, y_pos_est, y_pos) :
        loss = self.loss_fn(td_estimate, td_target) + (0.5 * self.loss_fn(y_pos_est, y_pos))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def sync_Q_target(self):
        self.net.target_features.load_state_dict(self.net.online_features.state_dict())
        self.net.target_td_est.load_state_dict(self.net.online_td_est.state_dict())


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
        state, y_pos, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est, y_pos_est = self.td_and_aux_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt, y_pos_est, torch.reshape(y_pos, y_pos_est.shape))

        return (td_est.mean().item(), loss)


class MarioNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 32, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 32, got: {w}")

        #       self.online = nn.Sequential(
        #           nn.Conv2d(in_channels=c, out_channels=32, kernel_size=5, stride=2, padding=1),
        #           nn.ReLU(),
        #           nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
        #           nn.ReLU(),
        #           #nn.MaxPool2d(3),
                
        #           nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
        #           nn.ReLU(),
        #           nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
        #           nn.MaxPool2d(2),
                
        #           nn.Flatten(),
        #           nn.Linear(18496, 128),
        #           nn.ReLU(),
        #           nn.Linear(128, output_dim)
        #       )

        self.online_features = self.feature_extraction(c, h, w, output_dim)
        self.online_td_est  = self.td_est(output_dim)
        self.online_aux = self.y_pos_est()

        self.target_features = copy.deepcopy(self.online_features)
        self.target_td_est = copy.deepcopy(self.online_td_est)
            
        # Q_target parameters are frozen.
        for p in self.target_features.parameters():
            p.requires_grad = False
        for p in self.target_td_est.parameters():
            p.requires_grad = False

        
    def feature_extraction(self, c, h, w, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            #nn.MaxPool2d(3),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

    def td_est(self, output_dim):
        return nn.Sequential(
            nn.Linear(18496, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def y_pos_est(self):
        return nn.Sequential(
            nn.Linear(18496, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, input, model):
        if model == 'online':
            features = self.online_features(input)
            return self.online_td_est(features), self.online_aux(features)
        elif model == 'target':
            features = self.target_features(input)
            return self.target_td_est(features), []

