#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 02:38:44 2019

@author: ccyen
"""

import numpy as np

def queue_backpressure_opt(Queue_length, Saturation_rates): # return 1, 2, 3, or 4
    phase = 4
    
    pressure = np.array([0 for i in range(phase)])
    
    pressure[0] = Queue_length[2]*Saturation_rates[2] + Queue_length[6]*Saturation_rates[6] # phase1: 3, 7
    pressure[1] = Queue_length[3]*Saturation_rates[3] + Queue_length[7]*Saturation_rates[7] # phase2: 4, 8
    pressure[2] = Queue_length[0]*Saturation_rates[0] + Queue_length[4]*Saturation_rates[4] # phase3: 1, 5
    pressure[3] = Queue_length[1]*Saturation_rates[1] + Queue_length[5]*Saturation_rates[5] # phase4: 2, 6
    
    value = pressure.max(0)
    opt_phase = pressure.argmax(0)
    
    return opt_phase + 1

def delay_backpressure_opt(HOL_wait_time, Saturation_rates): # return 1, 2, 3, or 4
    phase = 4
    
    pressure = np.array([0. for i in range(phase)])
    
    pressure[0] = HOL_wait_time[2]*Saturation_rates[2] + HOL_wait_time[6]*Saturation_rates[6] # phase1: 3, 7
    pressure[1] = HOL_wait_time[3]*Saturation_rates[3] + HOL_wait_time[7]*Saturation_rates[7] # phase2: 4, 8
    pressure[2] = HOL_wait_time[0]*Saturation_rates[0] + HOL_wait_time[4]*Saturation_rates[4] # phase3: 1, 5
    pressure[3] = HOL_wait_time[1]*Saturation_rates[1] + HOL_wait_time[5]*Saturation_rates[5] # phase4: 2, 6
    
    value = pressure.max(0)
    opt_phase = pressure.argmax(0)
    
    return opt_phase + 1

def weighted_backpressure_opt(Queue_length, HOL_wait_time, Saturation_rates): # return 1, 2, 3, or 4
    phase = 4
    
    pressure = np.array([0 for i in range(phase)])
    
    r = 100
    gamma_d = 1 / (1 + r)  
    gamma_q = r / (1 + r) 
    #gamma_d = 0.1  gamma_q = 0.9 # r = 9
    #gamma_d = 0.02   gamma_q = 0.98 # r = 49
    #gamma_d = 0.01  gamma_q = 0.99 # r = 99
    #gamma_d = 0.001  gamma_q = 0.999 # r = 999
    
    pressure[0] = (gamma_d * HOL_wait_time[2] + gamma_q * Queue_length[2])*Saturation_rates[2] + (gamma_d * HOL_wait_time[6] + gamma_q * Queue_length[6])*Saturation_rates[6] # phase1: 3, 7
    pressure[1] = (gamma_d * HOL_wait_time[3] + gamma_q * Queue_length[3])*Saturation_rates[3] + (gamma_d * HOL_wait_time[7] + gamma_q * Queue_length[7])*Saturation_rates[7] # phase2: 4, 8
    pressure[2] = (gamma_d * HOL_wait_time[0] + gamma_q * Queue_length[0])*Saturation_rates[0] + (gamma_d * HOL_wait_time[4] + gamma_q * Queue_length[4])*Saturation_rates[4] # phase3: 1, 5
    pressure[3] = (gamma_d * HOL_wait_time[1] + gamma_q * Queue_length[1])*Saturation_rates[1] + (gamma_d * HOL_wait_time[5] + gamma_q * Queue_length[5])*Saturation_rates[5] # phase4: 2, 6
    
    value = pressure.max(0)
    opt_phase = pressure.argmax(0)
    
    return opt_phase + 1

def two_level_opt(HOL_wait_time, Saturation_rates, act_w): # return 1, 2, 3, or 4
    phase = 4
    
    pressure = np.array([0. for i in range(phase)])
    
    pressure[0] = (HOL_wait_time[2]*Saturation_rates[2] + HOL_wait_time[6]*Saturation_rates[6]) * act_w[0] # phase1: 3, 7
    pressure[1] = (HOL_wait_time[3]*Saturation_rates[3] + HOL_wait_time[7]*Saturation_rates[7]) * 0 # phase2: 4, 8
    pressure[2] = (HOL_wait_time[0]*Saturation_rates[0] + HOL_wait_time[4]*Saturation_rates[4]) * act_w[1] # phase3: 1, 5
    pressure[3] = (HOL_wait_time[1]*Saturation_rates[1] + HOL_wait_time[5]*Saturation_rates[5]) * 0 # phase4: 2, 6
    
    value = pressure.max(0)
    opt_phase = pressure.argmax(0)
    
    return opt_phase + 1
