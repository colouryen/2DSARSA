#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 22:54:04 2019

@author: ccyen
"""

import array as arr
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from Jain_fairness_index import Jain_fairness_index


phase_to_server = [[3, 7], [4, 8], [1, 5], [2, 6]]

outflow_table = [[1, 1], [2, 8], [3, 3], [4, 2], [5, 5], [6, 4], [7, 7], [8, 6]]


def initiate(m, numOfLanes, t, T, Arrival_times_drl, Queue_length_mat_drl, System_states_drl):
    HOL_wait_time = np.zeros((m * m, numOfLanes))
    
    for x in range(1, m * m + 1, 1):
        """"" INITIAL STATES """""
        for i in range(1, numOfLanes + 1, 2):
            System_states_drl[x][i] = np.extract(Arrival_times_drl[x][i] <= t, Arrival_times_drl[x][i])
            Queue_length_mat_drl[x][i - 1, int(t/T) - 1] = np.size(System_states_drl[x][i])
            
            if np.size(System_states_drl[x][i]) > 0: # not empty
                HOL_wait_time[x - 1, i - 1] = t - System_states_drl[x][i][0]
    
    return HOL_wait_time
    
def run(m, numOfLanes, t, T, Arrival_times_drl, Departure_times_drl, Queue_length_mat_drl, System_states_drl, Saturation_rates, Policy):
    HOL_wait_time = np.zeros((m * m, numOfLanes))
    totalServedVehicles = 0
    end_to_end_ServedVehicles = 0
    totalArrivals = 0
    '''totalDelay_drl = dict.fromkeys(range(1, m * m + 1), )'''
    totalDelay_DRL = np.array([])
    Served_veh_count = 0 
    reward = 0
    
    for x in range(1, m * m + 1, 1):
        '''totalDelay_drl[x] = dict.fromkeys(range(1, numOfLanes + 1), )'''
        
        """"" UPDATE STATES """""
        for i in range(1, numOfLanes + 1, 2):
            This_slot_arrivals = np.extract((Arrival_times_drl[x][i] <= t) & (Arrival_times_drl[x][i] > t - T), Arrival_times_drl[x][i])
            
            New_arrivals = np.concatenate([System_states_drl[x][i], This_slot_arrivals])

            Served_last_slot = 0

            phase_of_last_slot = Policy[x - 1, int(t/T) - 1] # 1 or 2 or 3 or 4

            if (i == phase_to_server[int(phase_of_last_slot - 1)][0]) or (i == phase_to_server[int(phase_of_last_slot - 1)][1]): #scheduled servers
                R_max_serve = Saturation_rates[x - 1, i - 1] * T
                Served_last_slot = np.floor(R_max_serve * (1 - np.exp(-np.size(New_arrivals)/R_max_serve))) # floor to integer, right??

                Served_last_slot = int(min(np.size(New_arrivals), Served_last_slot))
                
                Served_veh_count += Served_last_slot;
                
                #if Served_last_slot > 0:
                # vehicles that outflow to other intersections
                # assume time + exp arrivals to the next intersection
                #newComingVehicles = np.full((1, int(Served_last_slot)), (t + T))
                newComingVehicles = np.array([(t + T) for i_1 in range(int(Served_last_slot))])
                lam = 0.0
                if Served_last_slot > 0: 
                    lam = 1.0/(float(Served_last_slot)/float(T))
                con_intervals = np.random.exponential(lam, (1, Served_last_slot)) 
                newComingVehicles = np.add(newComingVehicles, np.cumsum(con_intervals))
                
                newPaddingZero = np.array([0 for i_2 in range(int(Served_last_slot))]) 
                
                
                if (i == 3) and (((x - 1) % m) - 1 >= 0): # y - 1 > 0
                    Arrival_times_drl[x - 1][outflow_table[i - 1][1]] = np.concatenate([Arrival_times_drl[x - 1][outflow_table[i - 1][1]], newComingVehicles])
                    Departure_times_drl[x - 1][outflow_table[i - 1][1]] = np.concatenate([Departure_times_drl[x - 1][outflow_table[i - 1][1]], newPaddingZero])
                elif (i == 7) and (((x - 1) % m) + 1 < m): # y + 1 <= m
                    Arrival_times_drl[x + 1][outflow_table[i - 1][1]] = np.concatenate([Arrival_times_drl[x + 1][outflow_table[i - 1][1]], newComingVehicles])
                    Departure_times_drl[x + 1][outflow_table[i - 1][1]] = np.concatenate([Departure_times_drl[x + 1][outflow_table[i - 1][1]], newPaddingZero])
                elif (i == 5) and (((x - 1) - m) >= 0): # x - 1 > 0
                    Arrival_times_drl[x - m][outflow_table[i - 1][1]] = np.concatenate([Arrival_times_drl[x - m][outflow_table[i - 1][1]], newComingVehicles])
                    Departure_times_drl[x - m][outflow_table[i - 1][1]] = np.concatenate([Departure_times_drl[x - m][outflow_table[i - 1][1]], newPaddingZero])
                elif (i == 1) and (((x - 1) + m) < m * m): # x + 1 <= m
                    Arrival_times_drl[x + m][outflow_table[i - 1][1]] = np.concatenate([Arrival_times_drl[x + m][outflow_table[i - 1][1]], newComingVehicles])
                    Departure_times_drl[x + m][outflow_table[i - 1][1]] = np.concatenate([Departure_times_drl[x + m][outflow_table[i - 1][1]], newPaddingZero])
               
                tmp = np.where(Departure_times_drl[x][i] == 0)
                if np.size(tmp) > 0:
                    begin = int(tmp[0][0])
                else:
                    begin = 0
                Departure_times_drl[x][i][begin : begin + Served_last_slot] = t #[t] * (Served_last_slot)
                
                if (x in {1, 4, 7}) and (i == 3):
                    t_DRL = Departure_times_drl[x][i][begin : begin + Served_last_slot] - Arrival_times_drl[x + 2][i][begin : begin + Served_last_slot]
                    totalDelay_DRL = np.concatenate([totalDelay_DRL, t_DRL])
                    end_to_end_ServedVehicles = end_to_end_ServedVehicles + Served_last_slot
                elif (x in {3, 6, 9}) and (i == 7):
                    t_DRL = Departure_times_drl[x][i][begin : begin + Served_last_slot] - Arrival_times_drl[x - 2][i][begin : begin + Served_last_slot]
                    totalDelay_DRL = np.concatenate([totalDelay_DRL, t_DRL])
                    end_to_end_ServedVehicles = end_to_end_ServedVehicles + Served_last_slot
                elif (x in {7, 8, 9}) and (i == 1):
                    t_DRL = Departure_times_drl[x][i][begin : begin + Served_last_slot] - Arrival_times_drl[x - 6][i][begin : begin + Served_last_slot]
                    totalDelay_DRL = np.concatenate([totalDelay_DRL, t_DRL])
                    end_to_end_ServedVehicles = end_to_end_ServedVehicles + Served_last_slot
                elif (x in {1, 2, 3}) and (i == 5):
                    t_DRL = Departure_times_drl[x][i][begin : begin + Served_last_slot] - Arrival_times_drl[x + 6][i][begin : begin + Served_last_slot]
                    totalDelay_DRL = np.concatenate([totalDelay_DRL, t_DRL]) 
                    end_to_end_ServedVehicles = end_to_end_ServedVehicles + Served_last_slot
                    
                totalArrivals = totalArrivals + np.size(This_slot_arrivals)
                totalServedVehicles = totalServedVehicles + Served_last_slot
            
            System_states_drl[x][i] = New_arrivals[Served_last_slot:]
            Queue_length_mat_drl[x][i - 1, int(t/T) - 1] = np.size(System_states_drl[x][i])

            if np.size(System_states_drl[x][i]) > 0: # not empty
                HOL_wait_time[x - 1, i - 1] = t - System_states_drl[x][i][0]
            '''
            leaved_index = np.where(Departure_times_drl[x][i] > 0)
            totalDelay_drl[x][i] = Departure_times_drl[x][i][leaved_index] - Arrival_times_drl[x][i][leaved_index]
            totalDelay_DRL = np.concatenate([totalDelay_DRL, totalDelay_drl[x][i]])
            '''
            
    # Just maximizing throughput as reward function
    #if np.size(totalDelay_DRL) > 0:
    #    reward = end_to_end_ServedVehicles
    #else:
    #    reward = 0  
    
    # Average end-to-end delay as reward function
    #if np.size(totalDelay_DRL) > 0:
    #    reward = -1 * np.mean(totalDelay_DRL)
    #else:
    #    reward = 0
    
    # Jain fairness index as reward function
    #if np.size(totalDelay_DRL) > 0:
    #    reward = Jain_fairness_index(totalDelay_DRL)
    #else:
    #    reward = 0  
    
    # Power metric as reward function
    if np.size(totalDelay_DRL) > 0:
        reward = end_to_end_ServedVehicles/np.mean(totalDelay_DRL)
    else:
        reward = 0  
    
    # Power metric over 1 - Jain fairness index as reward function
    #if np.size(totalDelay_DRL) > 0 and Jain_fairness_index(totalDelay_DRL) < 1:
    #    power = totalServedVehicles / np.mean(totalDelay_DRL)
    #    fairness = Jain_fairness_index(totalDelay_DRL)
    #    reward = power / (1 - fairness)
    #else:
    #    reward = 0
    
    # Power metric times Jain fairness index as reward function
    #if np.size(totalDelay_DRL) > 0:
    #    power = totalServedVehicles / np.mean(totalDelay_DRL)
    #    fairness = Jain_fairness_index(totalDelay_DRL)
    #    reward = power * fairness
    #else:
    #    reward = 0  
    
    return HOL_wait_time, reward

##### Initialize arrivals #####

def arrivals_generator(lam_vec, T_tot, numOfLanes):
    
    N_initial_vec = lam_vec * T_tot
    Arrival_times = dict.fromkeys(range(1, numOfLanes + 1), )
    for s in range(1, numOfLanes + 1, 1):
        # N = floor(N_initial_vec(s))
        N = int(np.floor(N_initial_vec[s - 1]))
    #     if (s == 1 || s == 5)
    #       Intervals = random(dis_vec{s}, lam_vec(s), 1, N) + 1;
    #     else
        Intervals = np.random.exponential(1/lam_vec[s - 1], (1, N))
    #     end
        Arrivals = np.cumsum(Intervals)
        stop = Arrivals[-1]
        
        while stop < T_tot:
            N = 2 * N
    #         if (s == 1 || s == 5)
    #           Intervals = random(dis_vec{s}, lam_vec(s), 1, N) + 1;
    #         else
            Intervals = np.random.exponential(1/lam_vec[s - 1], (1, N))
    #         end
            Arrivals = np.cumsum(Intervals)
            stop = Arrivals[-1]
            
        Arrivals = np.extract(Arrivals <= T_tot, Arrivals)
        Arrival_times[s] = Arrivals
    
    return Arrival_times


def multi_arrivals_generator(m, lam_vec, T_tot, numOfLanes):
    
    N_initial_vec = lam_vec
    Arrival_times = dict.fromkeys(range(1, m * m + 1), )
    
    for k in range(1, m * m + 1, 1):
        Arrival_times[k] = dict.fromkeys(range(1, numOfLanes + 1), )
        
        for s in range(1, numOfLanes + 1, 1):
            N = int(np.floor(N_initial_vec[k - 1, s - 1] * T_tot))
            Intervals = np.random.exponential(1/lam_vec[k - 1, s - 1], (1, N))
            Arrivals = np.cumsum(Intervals)
            stop = Arrivals[-1]
            
            while stop < T_tot: 
                N = 2 * N
                Intervals = np.random.exponential(1/lam_vec[k - 1, s - 1], (1, N))
                Arrivals = np.cumsum(Intervals)
                stop = Arrivals[-1]
            
            Arrivals = np.extract(Arrivals <= T_tot, Arrivals)
            #Arrivals = np.take(Arrivals, np.nonzero(Arrivals <= T_tot))
            #Arrival_times[k][s] = Arrivals;
            
            Arrivals_small = np.extract(Arrivals <= 300, Arrivals)
            
            if (s == 1 or s == 2) and (k <= m):
                Arrival_times[k][s] = Arrivals
            elif (s == 3 or s == 4) and (k % m == 0):
                Arrival_times[k][s] = Arrivals
            elif (s == 5 or s == 6) and (k > m * (m - 1)):
                Arrival_times[k][s] = Arrivals
            elif (s == 7 or s == 8) and (k % m == 1):
                Arrival_times[k][s] = Arrivals
            else:
                Arrival_times[k][s] = Arrivals_small
        
    return Arrival_times

def multi_1357_arrivals_generator(m, lam_vec, T_tot, numOfLanes):
    
    N_initial_vec = lam_vec
    Arrival_times = dict.fromkeys(range(1, m * m + 1), )
    
    for k in range(1, m * m + 1, 1):
        Arrival_times[k] = dict.fromkeys(range(1, numOfLanes + 1), )
        
        for s in range(1, numOfLanes + 1, 1):
            N = int(np.floor(N_initial_vec[k - 1, s - 1] * T_tot))
            Intervals = np.random.exponential(1/lam_vec[k - 1, s - 1], (1, N))
            Arrivals = np.cumsum(Intervals)
            stop = Arrivals[-1]
            
            while stop < T_tot: 
                N = 2 * N
                Intervals = np.random.exponential(1/lam_vec[k - 1, s - 1], (1, N))
                Arrivals = np.cumsum(Intervals)
                stop = Arrivals[-1]
            
            Arrivals = np.extract(Arrivals <= T_tot, Arrivals)
            #Arrivals = np.take(Arrivals, np.nonzero(Arrivals <= T_tot))
            #Arrival_times[k][s] = Arrivals;
            
            Arrivals_small = np.extract(Arrivals <= 300, Arrivals)
            
            if (s == 1) and (k <= m):
                Arrival_times[k][s] = Arrivals
            elif (s == 3) and (k % m == 0):
                Arrival_times[k][s] = Arrivals
            elif (s == 5) and (k > m * (m - 1)):
                Arrival_times[k][s] = Arrivals
            elif (s == 7) and (k % m == 1):
                Arrival_times[k][s] = Arrivals
            else:
                Arrival_times[k][s] = np.array([])#Arrivals_small
        
    return Arrival_times


##### Flow map generation #####

def flowMap_generator(m, numOfLanes, HOL_wait_time, diffWait_table):
    
    matOfIntersection = dict.fromkeys(range(1, m * m + 1), )
    
    baseSize = int(numOfLanes / 2) + 1
    
    for s in range(1, m * m + 1, 1):
        matOfIntersection[s] = np.zeros((int(numOfLanes / 2) + 1, int(numOfLanes / 2) + 1))
        
        matOfIntersection[s][1, 2] = HOL_wait_time[s - 1, 0]
        matOfIntersection[s][1, 3] = HOL_wait_time[s - 1, 1]
        matOfIntersection[s][2, 3] = HOL_wait_time[s - 1, 2]
        matOfIntersection[s][3, 3] = HOL_wait_time[s - 1, 3]
        matOfIntersection[s][3, 2] = HOL_wait_time[s - 1, 4]
        matOfIntersection[s][3, 1] = HOL_wait_time[s - 1, 5]
        matOfIntersection[s][2, 1] = HOL_wait_time[s - 1, 6]
        matOfIntersection[s][1, 1] = HOL_wait_time[s - 1, 7]
        matOfIntersection[s][2, 2] = np.sum(HOL_wait_time[s - 1, :]) / numOfLanes
        
        for i in range(1, numOfLanes + 1, 1):
            if i == 1:
                if((s - 1) - m >= 0):
                    matOfIntersection[s][0, 2] = HOL_wait_time[s - 1, i - 1] - HOL_wait_time[(s - 1) - m, diffWait_table[i - 1][1] - 1]
                else:
                    matOfIntersection[s][0, 2] = HOL_wait_time[s - 1, i - 1]
            elif i == 2:
                if((s - 1) - m >= 0):
                    matOfIntersection[s][0, 3] = HOL_wait_time[s - 1, i - 1] - HOL_wait_time[(s - 1) - m, diffWait_table[i - 1][1] - 1]
                else:
                    matOfIntersection[s][0, 3] = HOL_wait_time[s - 1, i - 1]
            elif i == 3:
                if (((s - 1) % m) + 1 < m):
                    matOfIntersection[s][2, 4] = HOL_wait_time[s - 1, i - 1] - HOL_wait_time[(s - 1) + 1, diffWait_table[i - 1][1] - 1]
                else:
                    matOfIntersection[s][2, 4] = HOL_wait_time[s - 1, i - 1]
            elif i == 4:
                if (((s - 1) % m) + 1 < m):
                    matOfIntersection[s][3, 4] = HOL_wait_time[s - 1, i - 1] - HOL_wait_time[(s - 1) + 1, diffWait_table[i - 1][1] - 1]
                else:
                    matOfIntersection[s][3, 4] = HOL_wait_time[s - 1, i - 1]
            elif i == 5:
                if (((s - 1) + m) < m * m):
                    matOfIntersection[s][4, 2] = HOL_wait_time[s - 1, i - 1] - HOL_wait_time[(s - 1) + m, diffWait_table[i - 1][1] - 1]
                else:
                    matOfIntersection[s][4, 2] = HOL_wait_time[s - 1, i - 1]
            elif i == 6:
                if (((s - 1) + m) < m * m):
                    matOfIntersection[s][4, 1] = HOL_wait_time[s - 1, i - 1] - HOL_wait_time[(s - 1) + m, diffWait_table[i - 1][1] - 1]
                else:
                    matOfIntersection[s][4, 1] = HOL_wait_time[s - 1, i - 1]
            elif i == 7:
                if (((s - 1) % m) - 1 >= 0):
                    matOfIntersection[s][2, 0] = HOL_wait_time[s - 1, i - 1] - HOL_wait_time[(s - 1) - 1, diffWait_table[i - 1][1] - 1]
                else:
                    matOfIntersection[s][2, 0] = HOL_wait_time[s - 1, i - 1]
            elif i == 8:
                if (((s - 1) % m) - 1 >= 0):
                    matOfIntersection[s][1, 0] = HOL_wait_time[s - 1, i - 1] - HOL_wait_time[(s - 1) - 1, diffWait_table[i - 1][1] - 1]
                else:
                    matOfIntersection[s][1, 0] = HOL_wait_time[s - 1, i - 1]
    
    
    h_dir = dict.fromkeys(range(1, m + 1), )
    for x in range(1, m + 1, 1):
        h_dir[x] = np.concatenate([matOfIntersection[i] for i in range(int(m * (x - 1) + 1), int(m * x + 1), 1)], axis=1) 
        
    fMap = np.concatenate([h_dir[y] for y in range(1, m + 1)], axis=0)
    
    ''' delat hat W '''
    for z in range(1, m * m + 1, 1):
        for i in range(1, numOfLanes + 1, 1):
            if i == 1:
                if((z - 1) - m >= 0):
                    fMap[0 + baseSize * int((z - 1) / m), 0 + baseSize * ((z - 1) % m)] = \
                    fMap[0 + baseSize * (int((z - 1) / m) - 1), 2 + baseSize * ((z - 1) % m)] - \
                    fMap[0 + baseSize * int((z - 1) / m), 2 + baseSize * ((z - 1) % m)]
                else:
                    fMap[0 + baseSize * int((z - 1) / m), 0 + baseSize * ((z - 1) % m)] = \
                    fMap[0 + baseSize * int((z - 1) / m), 2 + baseSize * ((z - 1) % m)]
            elif i == 2:
                if((z - 1) - m >= 0):
                    fMap[0 + baseSize * int((z - 1) / m), 1 + baseSize * ((z - 1) % m)] = \
                    fMap[3 + baseSize * (int((z - 1) / m) - 1), 4 + baseSize * ((z - 1) % m)] - \
                    fMap[0 + baseSize * int((z - 1) / m), 3 + baseSize * ((z - 1) % m)]
                else:
                    fMap[0 + baseSize * int((z - 1) / m), 1 + baseSize * ((z - 1) % m)] = \
                    fMap[0 + baseSize * int((z - 1) / m), 3 + baseSize * ((z - 1) % m)]
            elif i == 3:
                if (((z - 1) % m) + 1 < m):
                    fMap[0 + baseSize * int((z - 1) / m), 4 + baseSize * ((z - 1) % m)] = \
                    fMap[2 + baseSize * int((z - 1) / m), 4 + baseSize * (((z - 1) % m) + 1)] - \
                    fMap[2 + baseSize * int((z - 1) / m), 4 + baseSize * ((z - 1) % m)]
                else:
                    fMap[0 + baseSize * int((z - 1) / m), 4 + baseSize * ((z - 1) % m)] = \
                    fMap[0 + baseSize * int((z - 1) / m), 3 + baseSize * ((z - 1) % m)]
            elif i == 4:
                if (((z - 1) % m) + 1 < m):
                    fMap[1 + baseSize * int((z - 1) / m), 4 + baseSize * ((z - 1) % m)] = \
                    fMap[4 + baseSize * int((z - 1) / m), 1 + baseSize * (((z - 1) % m) + 1)] - \
                    fMap[3 + baseSize * int((z - 1) / m), 4 + baseSize * ((z - 1) % m)]
                else:
                    fMap[1 + baseSize * int((z - 1) / m), 4 + baseSize * ((z - 1) % m)] = \
                    fMap[3 + baseSize * int((z - 1) / m), 3 + baseSize * ((z - 1) % m)]           
            elif i == 5:
                if (((z - 1) + m) < m * m):
                    fMap[4 + baseSize * int((z - 1) / m), 4 + baseSize * ((z - 1) % m)] = \
                    fMap[4 + baseSize * (int((z - 1) / m) + 1), 2 + baseSize * ((z - 1) % m)] - \
                    fMap[4 + baseSize * int((z - 1) / m), 2 + baseSize * ((z - 1) % m)]
                else:
                    fMap[4 + baseSize * int((z - 1) / m), 4 + baseSize * ((z - 1) % m)] = \
                    fMap[4 + baseSize * int((z - 1) / m), 2 + baseSize * ((z - 1) % m)]
            elif i == 6:
                if (((z - 1) + m) < m * m):
                    fMap[4 + baseSize * int((z - 1) / m), 3 + baseSize * ((z - 1) % m)] = \
                    fMap[1 + baseSize * (int((z - 1) / m) + 1), 0 + baseSize * ((z - 1) % m)] - \
                    fMap[4 + baseSize * int((z - 1) / m), 1 + baseSize * ((z - 1) % m)]
                else:
                    fMap[4 + baseSize * int((z - 1) / m), 3 + baseSize * ((z - 1) % m)] = \
                    fMap[4 + baseSize * int((z - 1) / m), 1 + baseSize * ((z - 1) % m)]    
            elif i == 7:
                if (((z - 1) % m) - 1 >= 0):
                    fMap[4 + baseSize * int((z - 1) / m), 0 + baseSize * ((z - 1) % m)] = \
                    fMap[2 + baseSize * int((z - 1) / m), 0 + baseSize * (((z - 1) % m) - 1)] - \
                    fMap[2 + baseSize * int((z - 1) / m), 0 + baseSize * ((z - 1) % m)]
                else:
                    fMap[4 + baseSize * int((z - 1) / m), 0 + baseSize * ((z - 1) % m)] = \
                    fMap[2 + baseSize * int((z - 1) / m), 0 + baseSize * ((z - 1) % m)]
            elif i == 8:
                if (((z - 1) % m) - 1 >= 0):
                    fMap[3 + baseSize * int((z - 1) / m), 0 + baseSize * ((z - 1) % m)] = \
                    fMap[0 + baseSize * int((z - 1) / m), 3 + baseSize * (((z - 1) % m) - 1)] - \
                    fMap[1 + baseSize * int((z - 1) / m), 0 + baseSize * ((z - 1) % m)]
                else:
                    fMap[3 + baseSize * int((z - 1) / m), 0 + baseSize * ((z - 1) % m)] = \
                    fMap[1 + baseSize * int((z - 1) / m), 0 + baseSize * ((z - 1) % m)]
        
    return fMap    
  
def flowMap_1357_generator(m, numOfLanes, HOL_wait_time, diffWait_table):
    
    matOfIntersection = dict.fromkeys(range(1, m * m + 1), )
    
    baseSize = int(numOfLanes / 2) + 1
    
    for s in range(1, m * m + 1, 1):
        matOfIntersection[s] = np.zeros((baseSize, baseSize))
        
        matOfIntersection[s][1, 2] = HOL_wait_time[s - 1, 0]
        matOfIntersection[s][1, 3] = HOL_wait_time[s - 1, 1]
        matOfIntersection[s][2, 3] = HOL_wait_time[s - 1, 2]
        matOfIntersection[s][3, 3] = HOL_wait_time[s - 1, 3]
        matOfIntersection[s][3, 2] = HOL_wait_time[s - 1, 4]
        matOfIntersection[s][3, 1] = HOL_wait_time[s - 1, 5]
        matOfIntersection[s][2, 1] = HOL_wait_time[s - 1, 6]
        matOfIntersection[s][1, 1] = HOL_wait_time[s - 1, 7]
        matOfIntersection[s][2, 2] = np.sum(HOL_wait_time[s - 1, :]) / numOfLanes
        
        for i in range(1, numOfLanes + 1, 2):
            if i == 1:
                if((s - 1) - m >= 0):
                    matOfIntersection[s][0, 2] = HOL_wait_time[s - 1, i - 1] - HOL_wait_time[(s - 1) - m, diffWait_table[i - 1][1] - 1]
                else:
                    matOfIntersection[s][0, 2] = HOL_wait_time[s - 1, i - 1]
            elif i == 2:
                if((s - 1) - m >= 0):
                    matOfIntersection[s][0, 3] = HOL_wait_time[s - 1, i - 1] - HOL_wait_time[(s - 1) - m, diffWait_table[i - 1][1] - 1]
                else:
                    matOfIntersection[s][0, 3] = HOL_wait_time[s - 1, i - 1]
            elif i == 3:
                if (((s - 1) % m) + 1 < m):
                    matOfIntersection[s][2, 4] = HOL_wait_time[s - 1, i - 1] - HOL_wait_time[(s - 1) + 1, diffWait_table[i - 1][1] - 1]
                else:
                    matOfIntersection[s][2, 4] = HOL_wait_time[s - 1, i - 1]
            elif i == 4:
                if (((s - 1) % m) + 1 < m):
                    matOfIntersection[s][3, 4] = HOL_wait_time[s - 1, i - 1] - HOL_wait_time[(s - 1) + 1, diffWait_table[i - 1][1] - 1]
                else:
                    matOfIntersection[s][3, 4] = HOL_wait_time[s - 1, i - 1]
            elif i == 5:
                if (((s - 1) + m) < m * m):
                    matOfIntersection[s][4, 2] = HOL_wait_time[s - 1, i - 1] - HOL_wait_time[(s - 1) + m, diffWait_table[i - 1][1] - 1]
                else:
                    matOfIntersection[s][4, 2] = HOL_wait_time[s - 1, i - 1]
            elif i == 6:
                if (((s - 1) + m) < m * m):
                    matOfIntersection[s][4, 1] = HOL_wait_time[s - 1, i - 1] - HOL_wait_time[(s - 1) + m, diffWait_table[i - 1][1] - 1]
                else:
                    matOfIntersection[s][4, 1] = HOL_wait_time[s - 1, i - 1]
            elif i == 7:
                if (((s - 1) % m) - 1 >= 0):
                    matOfIntersection[s][2, 0] = HOL_wait_time[s - 1, i - 1] - HOL_wait_time[(s - 1) - 1, diffWait_table[i - 1][1] - 1]
                else:
                    matOfIntersection[s][2, 0] = HOL_wait_time[s - 1, i - 1]
            elif i == 8:
                if (((s - 1) % m) - 1 >= 0):
                    matOfIntersection[s][1, 0] = HOL_wait_time[s - 1, i - 1] - HOL_wait_time[(s - 1) - 1, diffWait_table[i - 1][1] - 1]
                else:
                    matOfIntersection[s][1, 0] = HOL_wait_time[s - 1, i - 1]
    
    
    h_dir = dict.fromkeys(range(1, m + 1), )
    for x in range(1, m + 1, 1):
        h_dir[x] = np.concatenate([matOfIntersection[i] for i in range(int(m * (x - 1) + 1), int(m * x + 1), 1)], axis=1) 
        
    fMap = np.concatenate([h_dir[y] for y in range(1, m + 1)], axis=0)
    
    ''' delat hat W '''
    for z in range(1, m * m + 1, 1):
        for i in range(1, numOfLanes + 1, 2):
            if i == 1:
                if((z - 1) - m >= 0):
                    fMap[0 + baseSize * int((z - 1) / m), 0 + baseSize * ((z - 1) % m)] = \
                    fMap[0 + baseSize * (int((z - 1) / m) - 1), 2 + baseSize * ((z - 1) % m)] - \
                    fMap[0 + baseSize * int((z - 1) / m), 2 + baseSize * ((z - 1) % m)]
                else:
                    fMap[0 + baseSize * int((z - 1) / m), 0 + baseSize * ((z - 1) % m)] = \
                    fMap[0 + baseSize * int((z - 1) / m), 2 + baseSize * ((z - 1) % m)]
            elif i == 2:
                if((z - 1) - m >= 0):
                    fMap[0 + baseSize * int((z - 1) / m), 1 + baseSize * ((z - 1) % m)] = \
                    fMap[3 + baseSize * (int((z - 1) / m) - 1), 4 + baseSize * ((z - 1) % m)] - \
                    fMap[0 + baseSize * int((z - 1) / m), 3 + baseSize * ((z - 1) % m)]
                else:
                    fMap[0 + baseSize * int((z - 1) / m), 1 + baseSize * ((z - 1) % m)] = \
                    fMap[0 + baseSize * int((z - 1) / m), 3 + baseSize * ((z - 1) % m)]
            elif i == 3:
                if (((z - 1) % m) + 1 < m):
                    fMap[0 + baseSize * int((z - 1) / m), 4 + baseSize * ((z - 1) % m)] = \
                    fMap[2 + baseSize * int((z - 1) / m), 4 + baseSize * (((z - 1) % m) + 1)] - \
                    fMap[2 + baseSize * int((z - 1) / m), 4 + baseSize * ((z - 1) % m)]
                else:
                    fMap[0 + baseSize * int((z - 1) / m), 4 + baseSize * ((z - 1) % m)] = \
                    fMap[0 + baseSize * int((z - 1) / m), 3 + baseSize * ((z - 1) % m)]
            elif i == 4:
                if (((z - 1) % m) + 1 < m):
                    fMap[1 + baseSize * int((z - 1) / m), 4 + baseSize * ((z - 1) % m)] = \
                    fMap[4 + baseSize * int((z - 1) / m), 1 + baseSize * (((z - 1) % m) + 1)] - \
                    fMap[3 + baseSize * int((z - 1) / m), 4 + baseSize * ((z - 1) % m)]
                else:
                    fMap[1 + baseSize * int((z - 1) / m), 4 + baseSize * ((z - 1) % m)] = \
                    fMap[3 + baseSize * int((z - 1) / m), 3 + baseSize * ((z - 1) % m)]           
            elif i == 5:
                if (((z - 1) + m) < m * m):
                    fMap[4 + baseSize * int((z - 1) / m), 4 + baseSize * ((z - 1) % m)] = \
                    fMap[4 + baseSize * (int((z - 1) / m) + 1), 2 + baseSize * ((z - 1) % m)] - \
                    fMap[4 + baseSize * int((z - 1) / m), 2 + baseSize * ((z - 1) % m)]
                else:
                    fMap[4 + baseSize * int((z - 1) / m), 4 + baseSize * ((z - 1) % m)] = \
                    fMap[4 + baseSize * int((z - 1) / m), 2 + baseSize * ((z - 1) % m)]
            elif i == 6:
                if (((z - 1) + m) < m * m):
                    fMap[4 + baseSize * int((z - 1) / m), 3 + baseSize * ((z - 1) % m)] = \
                    fMap[1 + baseSize * (int((z - 1) / m) + 1), 0 + baseSize * ((z - 1) % m)] - \
                    fMap[4 + baseSize * int((z - 1) / m), 1 + baseSize * ((z - 1) % m)]
                else:
                    fMap[4 + baseSize * int((z - 1) / m), 3 + baseSize * ((z - 1) % m)] = \
                    fMap[4 + baseSize * int((z - 1) / m), 1 + baseSize * ((z - 1) % m)]    
            elif i == 7:
                if (((z - 1) % m) - 1 >= 0):
                    fMap[4 + baseSize * int((z - 1) / m), 0 + baseSize * ((z - 1) % m)] = \
                    fMap[2 + baseSize * int((z - 1) / m), 0 + baseSize * (((z - 1) % m) - 1)] - \
                    fMap[2 + baseSize * int((z - 1) / m), 0 + baseSize * ((z - 1) % m)]
                else:
                    fMap[4 + baseSize * int((z - 1) / m), 0 + baseSize * ((z - 1) % m)] = \
                    fMap[2 + baseSize * int((z - 1) / m), 0 + baseSize * ((z - 1) % m)]
            elif i == 8:
                if (((z - 1) % m) - 1 >= 0):
                    fMap[3 + baseSize * int((z - 1) / m), 0 + baseSize * ((z - 1) % m)] = \
                    fMap[0 + baseSize * int((z - 1) / m), 3 + baseSize * (((z - 1) % m) - 1)] - \
                    fMap[1 + baseSize * int((z - 1) / m), 0 + baseSize * ((z - 1) % m)]
                else:
                    fMap[3 + baseSize * int((z - 1) / m), 0 + baseSize * ((z - 1) % m)] = \
                    fMap[1 + baseSize * int((z - 1) / m), 0 + baseSize * ((z - 1) % m)]
    
        
    return fMap    
       
    """
    for r in range(1, m * m + 1, 1):
        for i in range(1, numOfLanes + 1, 1):
            if (i == 1 or i == 2):
                if (((r - 1) + m) < m * m):
                    delta_W[r - 1, i - 1] = hat_W[r - 1, i - 1] - hat_W[(r - 1) + m, diffWait_table[i - 1][1] - 1]
                    #delta_W{r, v}(i) = max(delta_W{r, v}(i), 0);
                else:
                    delta_W[r - 1, i - 1] = hat_W[r - 1, i - 1];
            elif (i == 3 or i == 4):
                if (((r - 1) % m) - 1 >= 0):
                    delta_W[r - 1, i - 1] = hat_W[r - 1, i - 1] - hat_W[(r - 1) - 1, diffWait_table[i - 1][1] - 1]
                    #delta_W{r, v}(i) = max(delta_W{r, v}(i), 0);
                else:
                    delta_W[r - 1, i - 1] = hat_W[r - 1, i - 1]
            elif (i == 5 or i == 6):
                if (((r - 1) - m) >= 0):
                    delta_W[r - 1, i - 1] = hat_W[r - 1, i - 1] - hat_W[(r - 1) - m, diffWait_table[i - 1][1] - 1]
                    #delta_W{r, v}(i) = max(delta_W{r, v}(i), 0);
                else:
                    delta_W[r - 1, i - 1] = hat_W[r - 1, i - 1]
            elif (i == 7 or i == 8):
                if (((r - 1) % m) + 1 < m):
                    delta_W[r - 1, i - 1] = hat_W[r - 1, i - 1] - hat_W[(r - 1) + 1, diffWait_table[i - 1][1] - 1]
                    #delta_W{r, v}(i) = max(delta_W{r, v}(i), 0);
                else:
                    delta_W[r - 1, i - 1] = hat_W[r - 1, i - 1]
     """   
 