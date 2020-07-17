#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 19:30:31 2019

@author: ccyen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 12:55:29 2019

@author: ccyen
"""

import array as arr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp
import scipy.io as sio
import os.path as op
import copy as cp
import Backpressure as BP
import Traffic_Env as env

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import itertools
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import math
import warnings

from Jain_fairness_index import Jain_fairness_index
from utils.hyperparameters import Config
from torch.autograd import Variable
from ecdf import ecdf
from C3PO_Agent import Model_DQN, Model_N_Step_DQN, Model_DSARSA, Model_DSARSA_CA, Model_2DSARSA, Model_3DQN, Model_2DQN

warnings.filterwarnings("ignore", category=UserWarning)

m = 3 # the number of intersections = m by m

numOfLanes = 8   # lanes of each intersection

T = 5  # slot length
T_tot = 3600 #10 * 60 * 60   # 10 hours

phase_to_server = [[3, 7], [4, 8], [1, 5], [2, 6]]

outflow_table = [[1, 1], [2, 8], [3, 3], [4, 2], [5, 5], [6, 4], [7, 7], [8, 6]]

diffWait_table = [[1, 1], [2, 4], [3, 3], [4, 6], [5, 5], [6, 8], [7, 7], [8, 2]]


Saturation_rates = np.full((m * m, numOfLanes), 0.5)
lam_vec = np.ones((m * m, 1)) * np.multiply([0.2, 1, 0.5, 1, 0.2, 1, 1, 1], 0.125 * 1.5)
dis_vec = np.array([['exp'] * numOfLanes] * (m * m))


##### DQN Parameters #####
config = Config()

config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#epsilon variables
config.epsilon_start    = 1.0
config.epsilon_final    = 0.01
config.epsilon_decay    = 300 #30000
config.epsilon_by_time_slot = lambda ep: config.epsilon_final + (config.epsilon_start - config.epsilon_final) * math.exp(-1. * ep / config.epsilon_decay)

#misc agent variables
config.GAMMA = 0.99
config.LR    = 1e-4

#memory
config.TARGET_NET_UPDATE_FREQ = 1000 #1000
config.EXP_REPLAY_SIZE        = 300000 #100000
config.BATCH_SIZE             = 1024 #32

# For 3D
config.USE_NOISY_NETS=False
config.SIGMA_INIT=0.5
config.USE_PRIORITY_REPLAY=False
config.PRIORITY_ALPHA=0.6
config.PRIORITY_BETA_START=0.4
config.PRIORITY_BETA_FRAMES = 100000

#Learning control variables
config.LEARN_START = 100      #10000
config.MAX_EPISODES  = 1000   #1000000
config.UPDATE_FREQ = 1

#Nstep controls
config.N_STEPS = 3

#Define threshold
HomoThreshold = 170
HeterThreshold = 1045
DelayThreshold = HomoThreshold

#Define state space
shapeOfState = (1, m * 5, m * 5)

#Define action space
phase = '13'
sizeOfActions = len(list(itertools.product(phase, repeat=m * m)))
candidateOfAction = np.zeros((sizeOfActions, m * m))
ind = 0
for output in itertools.product(phase, repeat=m * m):
    for i in range(0, m * m, 1):
        candidateOfAction[ind, i] = output[i]
        
    ind += 1


"""""""""""""""""""""""""  Delay-based """""""""""""""""""""""""

model_name = "2DSARSA"
model = Model_2DSARSA(config=config, shapeOfState=shapeOfState, sizeOfActions=sizeOfActions)

cur_state = np.array([])
prev_state = np.array([])
cur_action = 0
prev_action = 0
R_history = []
Max_Delay_history = []
Avg_Delay_history = []
Fairness_history = []
Avg_queue_history_drl = []
minDelay = 10000

for episode in range(1, config.MAX_EPISODES + 1, 1):
    
    Arrival_times_drl = env.multi_1357_arrivals_generator(m, lam_vec, T_tot, numOfLanes)
    
    episode_reward = 0.0
    
    cur_fairness = 0.0
    prev_fairness = cur_fairness
    
    Departure_times_drl = dict.fromkeys(range(1, m * m + 1), )
    Queue_length_mat_drl = dict.fromkeys(range(1, m * m + 1), )
    System_states_drl = dict.fromkeys(range(1, m * m + 1), )
    
    for w in range(1, m * m + 1, 1):
        Queue_length_mat_drl[w] = np.zeros((numOfLanes, int(T_tot/T)))
        System_states_drl[w] = dict.fromkeys(range(1, numOfLanes + 1), )
        Departure_times_drl[w] = dict.fromkeys(range(1, numOfLanes + 1), )
        
        for s in range(1, numOfLanes + 1, 2):
            Departure_times_drl[w][s] = np.array([0 for i in range(np.size(Arrival_times_drl[w][s]))]) 
    
    
    Policy_vec_drl = np.zeros((m * m, int(T_tot/T + 1)))
    Avg_total_queue_length_drl = np.array([[0 for i in range(m)] for j in range(m)]) #np.zeros((m * m, 1))
    
    # update the vehicles present in the system at each optimization point
    # -----save their arrival times: the first is the HOL vehicle, 
    # the size is the queue length; when departures, just pop out the corresponding number 
    # of vehicles from the beginning, when arrivals, just
    # add from the last! Eight rows are 8 servers respectively.  
    
    HOL_wait_time = env.initiate(m, numOfLanes, 5, 5, Arrival_times_drl, Queue_length_mat_drl, System_states_drl)
    cur_state = env.flowMap_1357_generator(m, numOfLanes, HOL_wait_time, diffWait_table)
    
    for t in range(T + 5, T_tot + 1, T):
        
        epsilon = config.epsilon_by_time_slot(episode) 
        
        cur_action = model.get_action(cur_state, epsilon)
        
        prev_state = cur_state
        
        
        for a in range(1, m * m + 1, 1):
            """"" OPTIMAZATION & MAKE DECISION """""
            
            Policy_vec_drl[a - 1, (int(t/T) + 1) - 1] = candidateOfAction[cur_action, a - 1] # action for each intersection
            
        
        HOL_wait_time, reward = env.run(m, 
                                        numOfLanes, 
                                        t, 
                                        T, 
                                        Arrival_times_drl, 
                                        Departure_times_drl, 
                                        Queue_length_mat_drl, 
                                        System_states_drl, 
                                        Saturation_rates, 
                                        Policy_vec_drl)
        
        cur_state = env.flowMap_1357_generator(m, numOfLanes, HOL_wait_time, diffWait_table)
        
        
        ##### For D Sarsa #####
        model.update(prev_state, prev_action, reward, cur_state, cur_action, episode)
        
        
        episode_reward += reward
        
        prev_action = cur_action


    for b in range(1, m * m + 1, 1):
        Avg_total_queue_length_drl[int(np.floor((b - 1) / m)), (b - 1) % m] = np.sum(Queue_length_mat_drl[b])/(T_tot/T)
    
    Avg_queue_history_drl.append(np.sum(Avg_total_queue_length_drl))


    Delay_vec_DRL = np.array([])
    for inx in range(1, m * m + 1, 1):
        for lane in range(1, numOfLanes + 1, 2):
            Ddrl_1357 = np.array([])
            
            if (inx in {1, 4, 7}) and (lane == 3):
                sortedArrival_times_drl = np.sort(Arrival_times_drl[inx + 2][lane])
                finished_index = np.where(Departure_times_drl[inx][lane] > 0)
                Ddrl_1357 = Departure_times_drl[inx][lane][finished_index] - sortedArrival_times_drl[finished_index]
            elif (inx in {3, 6, 9}) and (lane == 7):
                sortedArrival_times_drl = np.sort(Arrival_times_drl[inx - 2][lane])
                finished_index = np.where(Departure_times_drl[inx][lane] > 0)
                Ddrl_1357 = Departure_times_drl[inx][lane][finished_index] - sortedArrival_times_drl[finished_index]
            elif (inx in {7, 8, 9}) and (lane == 1):
                sortedArrival_times_drl = np.sort(Arrival_times_drl[inx - 6][lane])
                finished_index = np.where(Departure_times_drl[inx][lane] > 0)
                Ddrl_1357 = Departure_times_drl[inx][lane][finished_index] - sortedArrival_times_drl[finished_index]
            elif (inx in {1, 2, 3}) and (lane == 5):
                sortedArrival_times_drl = np.sort(Arrival_times_drl[inx + 6][lane])
                finished_index = np.where(Departure_times_drl[inx][lane] > 0)
                Ddrl_1357 = Departure_times_drl[inx][lane][finished_index] - sortedArrival_times_drl[finished_index]
            
            Delay_vec_DRL = np.concatenate([Delay_vec_DRL, Ddrl_1357])


    avgD = np.mean(Delay_vec_DRL)
    Avg_Delay_history.append(avgD)

    fair = Jain_fairness_index(Delay_vec_DRL)
    Fairness_history.append(fair)

    """"" Save Data and Save history """""
    R_history.append(episode_reward)
    sio.savemat('DRL_1357_train_data_d.mat', {'R_history':R_history, 'Avg_Delay_history': Avg_Delay_history, 'Fairness_history': Fairness_history, 'Avg_queue_history_drl':Avg_queue_history_drl})


    print('Episode: %d' %episode)
    print('Epsilon: %f' %epsilon)
    print('Fairness: %.4f' %fair)
    print('Reward: %.2f' %episode_reward)
    print('Avg Queue: %d' %np.sum(Avg_total_queue_length_drl))


""" Save Learned Model """
model.save_w(model_name)
