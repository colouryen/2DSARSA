#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 01:25:54 2019

@author: ccyen
"""

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import gym, random, pickle, os.path, math, glob

from torch.autograd import Variable
from timeit import default_timer as timer
from datetime import timedelta
from timeit import default_timer as timer
from IPython.display import clear_output
from utils.wrappers import *
from utils.hyperparameters import Config
from utils.plot import plot_reward
from agents.BaseAgent import BaseAgent
from networks.layers import NoisyLinear

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DuelingDQN(nn.Module):
    def __init__(self, shapeOfState, numOfActions):
        super(DuelingDQN, self).__init__()
        
        self.shape_state = shapeOfState
        self.num_actions = numOfActions
        
        self.layer1 = nn.Conv2d(self.shape_state[0], 32, kernel_size=5, stride=1)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=4, stride=1)
        self.layer3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.adv1 = nn.Linear(self.feature_size(), 1024)
        self.adv2 = nn.Linear(1024, self.num_actions)

        self.val1 = nn.Linear(self.feature_size(), 1024)
        self.val2 = nn.Linear(1024, 1)
        
        print(self.feature_size())
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = x.view(x.size(0), -1)

        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv)

        val = F.relu(self.val1(x))
        val = self.val2(val)

        return val + adv - adv.mean()
    
    def get_hidden_layer(self, state):
        y = F.relu(self.layer1(state))
        y = F.relu(self.layer2(y))
        y = F.relu(self.layer3(y))
        y = y.view(y.size(0), -1)

        output_hidden = F.relu(self.adv1(y))
        
        return output_hidden
    
    def feature_size(self):
        return self.layer3(self.layer2(self.layer1(torch.zeros(1, *self.shape_state)))).view(1, -1).size(1)
    
    def sample_noise(self):
        #ignore this for now
        pass

class CNN(nn.Module):
    def __init__(self, shapeOfState, numOfActions):
        super(CNN, self).__init__()
        
        self.shape_state = shapeOfState
        self.num_actions = numOfActions
        #TO DO Declare your layers here
        self.layer1 = nn.Conv2d(self.shape_state[0], 32, kernel_size=5, stride=1)
        #self.layer2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=4, stride=1)
        self.layer3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        #self.layer4 = nn.Linear(self.feature_size(), self.num_actions)
        self.layer4 = nn.Linear(self.feature_size(), 1024)
        self.layer5 = nn.Linear(1024, self.num_actions)
        print(self.feature_size())

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = x.view(x.size(0), -1) # transform x to one dimension
        #x = self.layer4(x)
        x = F.relu(self.layer4(x))
        x = self.layer5(x)
        
        return x
    
    def get_hidden_layer(self, state):
        y = F.relu(self.layer1(state))
        y = F.relu(self.layer2(y))
        y = F.relu(self.layer3(y))
        y = y.view(y.size(0), -1)

        output_hidden = F.relu(self.layer4(y))
        
        return output_hidden
    
    def feature_size(self):
        return self.layer3(self.layer2(self.layer1(torch.zeros(1, *self.shape_state)))).view(1, -1).size(1)

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(self.feature_size(), 512)
        self.fc2 = nn.Linear(512, self.num_actions)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).size(1)
