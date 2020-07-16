#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 13:01:22 2019

@author: ccyen
"""

import numpy as np

def Jain_fairness_index(delay_vec):
    n = np.size(delay_vec)
    squ_sum = 0.0
    
    for i in range(n):
        squ_sum = squ_sum + delay_vec[i] * delay_vec[i]
    
    num = np.sum(delay_vec)**2/squ_sum/n
    
    return num
