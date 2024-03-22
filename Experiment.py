#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

from CartPoleDQN import dqn
from Helper import LearningCurvePlot, smooth
from Agent import DQNAgent

def average_over_repetitions(n_repetitions, n_timesteps, max_episode_length, use_replay_buffer, learning_rate, 
                                          gamma, policy, epsilon, epsilon_decay, epsilon_min, temp, temp_min, temp_decay, smoothing_window=None, eval_interval=500,batch_size=64):

    returns_over_repetitions = []
    now = time.time()
    
    for rep in range(n_repetitions): 
        
        returns, timesteps = dqn(n_timesteps, use_replay_buffer, learning_rate, gamma, policy, epsilon, temp,eval_interval,batch_size=batch_size)
        returns_over_repetitions.append(returns)
        print("Done nr: ", rep)
    print(returns_over_repetitions)
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))
    learning_curve = np.mean(np.array(returns_over_repetitions),axis=0) # average over repetitions  
    # if smoothing_window is not None: 
    #     learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve, timesteps  

def experiment():
    ####### Settings
    n_repetitions = 20
    smoothing_window = 9 # Must be an odd number. Use 'None' to switch smoothing off!
    use_replay_buffer = False
        
    n_timesteps = 50001 # Set one extra timestep to ensure evaluation at start and end
    eval_interval = 500
    max_episode_length = 100
    gamma = 0.99
    
    if use_replay_buffer:
        batch_size = 32
    else:
        batch_size = 1
    
    policies = ['softmax','egreedy'] # 'egreedy' or 'softmax' 
    epsilon = 0.1
    epsilon_min = 0.05
    epsilon_decay =0.995
    temp = 0.1
    temp_min = 0.01
    temp_decay = 0.995
    # Back-up & update
    learning_rate = 0.01
    
    Plot = LearningCurvePlot(title = "DQN-ER")    
    Plot.set_ylim(0, 600) 
    for policy in policies:
        learning_curve, timesteps = average_over_repetitions(n_repetitions=n_repetitions, n_timesteps=n_timesteps, max_episode_length=max_episode_length, use_replay_buffer = use_replay_buffer, learning_rate=learning_rate, 
                                          gamma=gamma, policy=policy, epsilon=epsilon, epsilon_decay=epsilon_decay , epsilon_min=epsilon_min, temp=temp, temp_min=temp_min, temp_decay=temp_decay, smoothing_window=smoothing_window, eval_interval=eval_interval,batch_size=batch_size)
        if policy == 'softmax':
            Plot.add_curve(timesteps,learning_curve,label=(str(policy)+ ", temp= "+str(temp)))
        elif policy == 'egreedy':
            Plot.add_curve(timesteps,learning_curve,label=(str(policy)+ ", eps= "+str(temp)))

    Plot.save('dqn_no_ER.png')

if __name__ == '__main__':
    experiment()

