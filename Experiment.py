#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

from CartPoleDQN import dqn
from Helper import LearningCurvePlot, smooth
from Agent import DQNAgent

def average_over_repetitions(n_repetitions, n_timesteps, max_episode_length, learning_rate, 
                                          gamma, policy, epsilon, epsilon_decay, epsilon_min, temp, smoothing_window=None, eval_interval=500,batch_size=64):

    returns_over_repetitions = []
    now = time.time()
    
    # dqn_agent_and_model = DQNAgent(n_states=4, 
    #                     n_actions=2, 
    #                     learning_rate=learning_rate, 
    #                     gamma=gamma,
    #                     epsilon=epsilon,
    #                     epsilon_decay=epsilon_decay,
    #                     epsilon_min=epsilon_min,
    #                     temp=temp)
    
    for rep in range(n_repetitions): 
        
        returns, timesteps = dqn(n_timesteps, learning_rate, gamma, policy, epsilon, temp, eval_interval)
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
        
    n_timesteps = 80001 # Set one extra timestep to ensure evaluation at start and end
    eval_interval = 500
    max_episode_length = 100
    gamma = 0.99
    batch_size = 32
    
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.001
    epsilon_min = 0.05
    epsilon_decay =0.995
    temp = 0.1
    # Back-up & update
    learning_rate = 0.001
    
    Plot = LearningCurvePlot(title = r'$\epsilon$-greedy')    
    Plot.set_ylim(0, 200) 
    learning_curve, timesteps = average_over_repetitions(n_repetitions=n_repetitions, n_timesteps=n_timesteps, max_episode_length=max_episode_length, learning_rate=learning_rate, 
                                          gamma=gamma, policy=policy, epsilon=epsilon, epsilon_decay=epsilon_decay , epsilon_min=epsilon_min, temp=temp, smoothing_window=smoothing_window, eval_interval=eval_interval,batch_size=batch_size)
    
    Plot.add_curve(timesteps,learning_curve,label=r'$\epsilon$-greedy, $\epsilon $ = {}'.format(epsilon))
    
    Plot.save('dqn_egreedy_epsilon0.001.png')

if __name__ == '__main__':
    experiment()

