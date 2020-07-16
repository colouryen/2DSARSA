#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 01:18:36 2019

@author: ccyen
"""

import gym, random, pickle, os.path, math, glob
import numpy as np
import itertools

from timeit import default_timer as timer
from datetime import timedelta
from timeit import default_timer as timer

import torch
import torch.optim as optim

import matplotlib
from IPython.display import clear_output

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utils.wrappers import *
from utils.hyperparameters import Config
from utils.plot import plot_reward

from C3PO_BaseAgent import BaseAgent
from C3PO_Network import CNN, CNN_Actor, CNN_Critic, DQN, DuelingDQN, BDN, ActorCritic, DuelingDQN_with_Noisy
from C3PO_RolloutStorage import RolloutStorage

from C3PO_MemReplay import ReplayBuffer, ExperienceReplayMemory, PrioritizedReplayMemory

class Model_3DQN(BaseAgent):
    def __init__(self, static_policy=False, config=None, shapeOfState=None, sizeOfActions=None):
        super(Model_3DQN, self).__init__(config=config, shapeOfState=shapeOfState, sizeOfActions=sizeOfActions)
        self.device = config.device

        #self.noisy=config.USE_NOISY_NETS
        self.priority_replay=config.USE_PRIORITY_REPLAY

        self.gamma = config.GAMMA
        self.lr = config.LR
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.learn_start = config.LEARN_START
        self.update_freq = config.UPDATE_FREQ
        self.sigma_init= config.SIGMA_INIT
        self.priority_beta_start = config.PRIORITY_BETA_START
        self.priority_beta_frames = config.PRIORITY_BETA_FRAMES
        self.priority_alpha = config.PRIORITY_ALPHA

        self.static_policy = static_policy
        self.num_feats = shapeOfState
        self.num_actions = sizeOfActions
        
        self.declare_networks()
            
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        #move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()

        self.update_count = 0

        self.declare_memory()

        self.nsteps = config.N_STEPS
        self.nstep_buffer = []

    def declare_networks(self):
        #self.model = DuelingDQN_with_Noisy(self.num_feats, self.num_actions, noisy=self.noisy, sigma_init=self.sigma_init)
        #self.target_model = DuelingDQN_with_Noisy(self.num_feats, self.num_actions, noisy=self.noisy, sigma_init=self.sigma_init)
        self.model = DuelingDQN(self.num_feats, self.num_actions)
        self.target_model = DuelingDQN(self.num_feats, self.num_actions)

    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size) if not self.priority_replay else PrioritizedReplayMemory(self.experience_replay_size, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames)

    def append_to_replay(self, s, a, r, s_):
        self.nstep_buffer.append((s, a, r, s_))

        if(len(self.nstep_buffer)<self.nsteps):
            return
        
        R = sum([self.nstep_buffer[i][2]*(self.gamma**i) for i in range(self.nsteps)])
        state, action, _, _ = self.nstep_buffer.pop(0)

        self.memory.push((state, action, R, s_))

    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (-1,)+self.num_feats

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        try: #sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights

    
    def compute_loss(self, batch_vars): #faster
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        #estimate
        #self.model.sample_noise()
        current_q_values = self.model(batch_state).gather(1, batch_action)
        
        #target
        with torch.no_grad():
            max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                #self.target_model.sample_noise()
                max_next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action)
            expected_q_values = batch_reward + ((self.gamma**self.nsteps)*max_next_q_values)

        diff = (expected_q_values - current_q_values)
        if self.priority_replay:
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())
            loss = self.MSE(diff).squeeze() * weights
        else:
            loss = self.MSE(diff)
        loss = loss.mean()

        return loss

    def update(self, s, a, r, s_, frame=0):
        if self.static_policy:
            return None

        self.append_to_replay(s, a, r, s_)

        if frame < self.learn_start or frame % self.update_freq != 0:
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_target_model()
        '''
        self.save_td(loss.item(), frame)
        self.save_sigma_param_magnitudes(frame)
        '''

    def get_action(self, s, eps=0.1): #faster
        with torch.no_grad():
            if np.random.random() >= eps or self.static_policy:# or self.noisy:
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                X = X.unsqueeze(0)
                #self.model.sample_noise()
                a = self.model(X).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    def update_target_model(self):
        self.update_count+=1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.model(next_states).max(dim=1)[1].view(-1, 1)

    def finish_nstep(self):
        while len(self.nstep_buffer) > 0:
            R = sum([self.nstep_buffer[i][2]*(self.gamma**i) for i in range(len(self.nstep_buffer))])
            state, action, _, _ = self.nstep_buffer.pop(0)

            self.memory.push((state, action, R, None))

    def reset_hx(self):
        pass

    def MSE(self, x):
        return 0.5 * x.pow(2)
    
class Model_2DQN(BaseAgent):
    def __init__(self, static_policy=False, config=None, shapeOfState=None, sizeOfActions=None):
        super(Model_2DQN, self).__init__(config=config, shapeOfState=shapeOfState, sizeOfActions=sizeOfActions)
        self.device = config.device
        
        self.gamma = config.GAMMA
        self.lr = config.LR
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.learn_start = config.LEARN_START
        self.update_freq = config.UPDATE_FREQ

        self.static_policy = static_policy
        self.num_feats = shapeOfState # 1 => gray scale, 3 => color scale
        self.num_actions = sizeOfActions
        
        self.declare_networks()
        
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        #move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()

        self.update_count = 0

        self.declare_memory()

    def declare_networks(self):
        self.model = DuelingDQN(self.num_feats, self.num_actions)
        self.target_model = DuelingDQN(self.num_feats, self.num_actions)
        #self.model = DQN(self.num_feats, self.num_actions)
        #self.target_model = DQN(self.num_feats, self.num_actions)
    
    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size)
    
    def append_to_replay(self, s, a, r, s_):
        self.memory.push((s, a, r, s_))
    
    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (-1,)+self.num_feats

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        try: #sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights
    
    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        #estimate
        current_q_values = self.model(batch_state).gather(1, batch_action)
        
        #target
        with torch.no_grad():
            max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                max_next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action)
            expected_q_values = batch_reward + self.gamma*max_next_q_values

        diff = (expected_q_values - current_q_values)
        loss = self.MSE(diff)
        loss = loss.mean()

        return loss

    def update(self, s, a, r, s_, epi):
        if self.static_policy:
            return None
        
        self.append_to_replay(s, a, r, s_)
        
        if ((epi < self.learn_start) or (epi % self.update_freq != 0)):
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        self.update_target_model()
        """
        self.save_td(loss.item(), epi)
        self.save_sigma_param_magnitudes(epi)
        """

    def get_action(self, s, eps=0.1):
        with torch.no_grad():
            if ((np.random.random() >= eps) or (self.static_policy)):
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                X = X.unsqueeze(0)
                a = self.model(X).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    def update_target_model(self):
        self.update_count+=1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def finish_nstep(self):
        pass

    def reset_hx(self):
        pass
    
    def MSE(self, x):
        return 0.5 * x.pow(2)

class Model_DQN(BaseAgent):
    def __init__(self, static_policy=False, config=None, shapeOfState=None, sizeOfActions=None):
        super(Model_DQN, self).__init__(config=config, shapeOfState=shapeOfState, sizeOfActions=sizeOfActions)
        self.device = config.device
        
        self.gamma = config.GAMMA
        self.lr = config.LR
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.learn_start = config.LEARN_START
        self.update_freq = config.UPDATE_FREQ

        self.static_policy = static_policy
        self.num_feats = shapeOfState # 1 => gray scale, 3 => color scale
        self.num_actions = sizeOfActions
        
        self.declare_networks()
        
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        #move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()

        self.update_count = 0

        self.declare_memory()

    def declare_networks(self):
        self.model = CNN(self.num_feats, self.num_actions)
        self.target_model = CNN(self.num_feats, self.num_actions)
        #self.model = DQN(self.num_feats, self.num_actions)
        #self.target_model = DQN(self.num_feats, self.num_actions)
    
    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size)
    
    def append_to_replay(self, s, a, r, s_):
        self.memory.push((s, a, r, s_))
    
    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (-1,)+self.num_feats

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        try: #sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights
    
    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        #estimate
        current_q_values = self.model(batch_state).gather(1, batch_action)
        
        #target
        with torch.no_grad():
            max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                max_next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action)
            expected_q_values = batch_reward + self.gamma*max_next_q_values

        diff = (expected_q_values - current_q_values)
        loss = self.MSE(diff)
        loss = loss.mean()

        return loss

    def update(self, s, a, r, s_, epi):
        if self.static_policy:
            return None
        
        self.append_to_replay(s, a, r, s_)
        
        if ((epi < self.learn_start) or (epi % self.update_freq != 0)):
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        self.update_target_model()
        """
        self.save_td(loss.item(), epi)
        self.save_sigma_param_magnitudes(epi)
        """

    def get_action(self, s, eps=0.1):
        with torch.no_grad():
            if ((np.random.random() >= eps) or (self.static_policy)):
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                X = X.unsqueeze(0)
                a = self.model(X).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    def update_target_model(self):
        self.update_count+=1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def finish_nstep(self):
        pass

    def reset_hx(self):
        pass
    
    def MSE(self, x):
        return 0.5 * x.pow(2)   

class Model_2DSARSA(BaseAgent):    
    def __init__(self, static_policy=False, config=None, shapeOfState=None, sizeOfActions=None):
        super(Model_2DSARSA, self).__init__(config=config, shapeOfState=shapeOfState, sizeOfActions=sizeOfActions)
        self.device = config.device

        self.priority_replay=config.USE_PRIORITY_REPLAY

        self.gamma = config.GAMMA
        self.lr = config.LR
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.learn_start = config.LEARN_START
        self.update_freq = config.UPDATE_FREQ
        self.priority_beta_start = config.PRIORITY_BETA_START
        self.priority_beta_frames = config.PRIORITY_BETA_FRAMES
        self.priority_alpha = config.PRIORITY_ALPHA

        self.static_policy = static_policy
        self.num_feats = shapeOfState
        self.num_actions = sizeOfActions

        self.declare_networks()
            
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        #move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()

        self.update_count = 0

        self.declare_memory()

    def declare_networks(self):
        self.model = DuelingDQN(self.num_feats, self.num_actions)
        self.target_model = DuelingDQN(self.num_feats, self.num_actions)

    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size) if not self.priority_replay else PrioritizedReplayMemory(self.experience_replay_size, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames)

    def append_to_replay(self, s, a, r, s_, a_):
        self.memory.push((s, a, r, s_, a_))
    
    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        batch_state, batch_action, batch_reward, batch_next_state, batch_next_action = zip(*transitions)

        shape = (-1,)+self.num_feats

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        try: #sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        batch_next_action = torch.tensor(batch_next_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)

        return batch_state, batch_action, batch_reward, non_final_next_states, batch_next_action, non_final_mask, empty_next_state_values, indices, weights
    
    def compute_loss(self, batch_vars): #faster
        batch_state, batch_action, batch_reward, non_final_next_states, batch_next_action, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        #estimate
        current_q_values = self.model(batch_state).gather(1, batch_action)
        
        #target
        with torch.no_grad():
            next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, batch_next_action)
            expected_q_values = batch_reward + self.gamma*next_q_values
        
        diff = (expected_q_values - current_q_values)
        if self.priority_replay:
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())
            loss = self.MSE(diff).squeeze() * weights
        else:
            loss = self.MSE(diff)
        loss = loss.mean()

        return loss

    def update(self, s, a, r, s_, a_, frame=0):
        if self.static_policy:
            return None

        self.append_to_replay(s, a, r, s_, a_)

        if frame < self.learn_start or frame % self.update_freq != 0:
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_target_model()
        
        '''
        self.save_td(loss.item(), frame)
        self.save_sigma_param_magnitudes(frame)
        '''

    def get_action(self, s, eps=0.1): #faster
        with torch.no_grad():
            if ((np.random.random() >= eps) or self.static_policy):
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                X = X.unsqueeze(0)
                a = self.model(X).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    def update_target_model(self):
        self.update_count+=1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def finish_nstep(self):
        pass
    
    def reset_hx(self):
        pass
    
    def MSE(self, x):
        return 0.5 * x.pow(2)

class Model_DSARSA(BaseAgent):
    def __init__(self, static_policy=False, config=None, shapeOfState=None, sizeOfActions=None):
        super(Model_DSARSA, self).__init__(config=config, shapeOfState=shapeOfState, sizeOfActions=sizeOfActions)
        self.device = config.device
        
        self.gamma = config.GAMMA
        self.lr = config.LR
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.learn_start = config.LEARN_START
        self.update_freq = config.UPDATE_FREQ

        self.static_policy = static_policy
        self.num_feats = shapeOfState # 1 => gray scale, 3 => color scale
        self.num_actions = sizeOfActions
        
        self.declare_networks()
        
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        #move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()

        self.update_count = 0

        self.declare_memory()

    def declare_networks(self):
        self.model = CNN(self.num_feats, self.num_actions)
        self.target_model = CNN(self.num_feats, self.num_actions)
        #self.model = DQN(self.num_feats, self.num_actions)
        #self.target_model = DQN(self.num_feats, self.num_actions)
    
    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size)
    
    def append_to_replay(self, s, a, r, s_, a_):
        self.memory.push((s, a, r, s_, a_))
    
    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        batch_state, batch_action, batch_reward, batch_next_state, batch_next_action = zip(*transitions)

        shape = (-1,)+self.num_feats

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        try: #sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True
        
        batch_next_action = torch.tensor(batch_next_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)

        return batch_state, batch_action, batch_reward, non_final_next_states, batch_next_action, non_final_mask, empty_next_state_values, indices, weights
    
    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, batch_next_action, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        #estimate
        current_q_values = self.model(batch_state).gather(1, batch_action)
        
        #target
        with torch.no_grad():
            next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, batch_next_action)
            expected_q_values = batch_reward + self.gamma*next_q_values

        diff = (expected_q_values - current_q_values)
        loss = self.MSE(diff)
        loss = loss.mean()

        return loss

    def update(self, s, a, r, s_, a_, epi):
        if self.static_policy:
            return None
        
        self.append_to_replay(s, a, r, s_, a_)
        
        if ((epi < self.learn_start) or (epi % self.update_freq != 0)):
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        self.update_target_model()
        """
        self.save_td(loss.item(), epi)
        self.save_sigma_param_magnitudes(epi)
        """

    def get_action(self, s, eps=0.1):
        with torch.no_grad():
            if ((np.random.random() >= eps) or (self.static_policy)):
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                X = X.unsqueeze(0)
                a = self.model(X).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    def update_target_model(self):
        self.update_count+=1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def finish_nstep(self):
        pass

    def reset_hx(self):
        pass
    
    def MSE(self, x):
        return 0.5 * x.pow(2)