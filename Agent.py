#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for master course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Helper import softmax, argmax
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from tensorflow.keras import models, layers, optimizers

class DQNAgent:
    """
        DQN Agent class, with e-greedy policy and experience replay buffer
    
    Raises:
        KeyError: Provide an epsilon
        KeyError: Provide a temperature

    Returns:
        int: best action according to the policy
    """
    def __init__(self, n_states, n_actions, learning_rate, gamma, epsilon=0.05, epsilon_decay=0.995, epsilon_min=0.01, temp=0.05):
        self.n_states = n_states
        self.n_actions = n_actions
        self.memory = deque(maxlen=2000) 
        self.gamma = gamma  
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.Q_sa = np.zeros((n_states,n_actions))
        self.model = self.build_model()
        
        print("DQNagent initialized with nr states: "+ str(n_states) + " and nr actions: " + str(n_actions))
        
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        if policy == 'greedy':
            '''Return greedy policy'''
            return argmax(self.Q_sa[s, :])
        elif policy == 'egreedy':
            '''Return e-greedy policy'''
            if epsilon is None:
                raise KeyError("Provide an epsilon")
            
            if np.random.rand() < epsilon:
                return np.random.randint(self.n_actions)
            else:
                greedy_action = np.argmax(self.Q_sa[s, :])
                probabilities = np.ones(self.n_actions) * (epsilon / self.n_actions)
                probabilities[greedy_action] += (1.0 - epsilon)
                return np.random.choice(np.arange(self.n_actions), p=probabilities)
            
        elif policy == 'softmax':
            '''Return Boltzmann (softmax) policy'''
            if temp is None:
                raise KeyError("Provide a temperature")
            
            q_values = self.Q_sa[s, :]
            a = np.argmax(softmax(q_values,temp=temp))
            return a

    def build_model(self):
        """Build neural network model
        
           In effect, the network is trying to predict the expected return 
           of taking each action given the current input.
        """
        # if GPU is to be used
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        super(DQNAgent, self).__init__()
        self.layer1 = nn.Linear(self.n_states, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, self.n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
    
    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = self.select_action(s, 'greedy')
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return


# class DQNAgent:

#     def __init__(self, n_states, n_actions, learning_rate, gamma):
#         self.n_states = n_states
#         self.n_actions = n_actions
#         self.learning_rate = learning_rate
#         self.gamma = gamma
#         self.Q_sa = np.zeros((n_states,n_actions))
        
#     def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
#         if policy == 'greedy':
#             '''Return greedy policy'''
#             return argmax(self.Q_sa[s, :])
#         elif policy == 'egreedy':
#             '''Return e-greedy policy'''
#             if epsilon is None:
#                 raise KeyError("Provide an epsilon")
            
#             if np.random.rand() < epsilon:
#                 return np.random.randint(self.n_actions)
#             else:
#                 greedy_action = np.argmax(self.Q_sa[s, :])
#                 probabilities = np.ones(self.n_actions) * (epsilon / self.n_actions)
#                 probabilities[greedy_action] += (1.0 - epsilon)
#                 return np.random.choice(np.arange(self.n_actions), p=probabilities)
            
#         elif policy == 'softmax':
#             '''Return Boltzmann (softmax) policy'''
#             if temp is None:
#                 raise KeyError("Provide a temperature")
            
#             q_values = self.Q_sa[s, :]
#             a = np.argmax(softmax(q_values,temp=temp))
#             return a
            
#     def update(self):
#         raise NotImplementedError('For each agent you need to implement its specific back-up method') # Leave this and overwrite in subclasses in other files


#     def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
#         returns = []  # list to store the reward per episode
#         for i in range(n_eval_episodes):
#             s = eval_env.reset()
#             R_ep = 0
#             for t in range(max_episode_length):
#                 a = self.select_action(s, 'greedy')
#                 s_prime, r, done = eval_env.step(a)
#                 R_ep += r
#                 if done:
#                     break
#                 else:
#                     s = s_prime
#             returns.append(R_ep)
#         mean_return = np.mean(returns)
#         return mean_return
