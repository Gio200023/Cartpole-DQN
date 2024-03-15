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
from ReplayBuffer import ReplayMemory
# from tensorflow.keras import models, layers, optimizers

class DQNAgent(nn.Module):
    """
        DQN Agent class, with e-greedy policy and experience replay buffer
    
    Raises:
        KeyError: Provide an epsilon
        KeyError: Provide a temperature

    Returns:
        int: best action according to the policy
    """
    def __init__(self, n_states, n_actions, learning_rate, gamma, epsilon=0.05, epsilon_decay=0.995, epsilon_min=0.01, temp=0.05):
        super(DQNAgent, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.replay_buffer = ReplayMemory(10000)
        self.gamma = gamma  
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.Q_sa = np.zeros((n_states,n_actions))
        
        
        # Network
        self.layer1 = nn.Linear(self.n_states, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, self.n_actions)
        # self.device = "cpu"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        #Hypertuning
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        state = np.array(s)
        state = torch.from_numpy(state).float().unsqueeze(0)
        
        state = state.to(self.device)
        
        # Epsilon-greedy policy
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                q_values = self.forward(state)
                return q_values.argmax().item()  # Return the action with the highest Q-value

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def replay(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        
        minibatch = random.sample(self.replay_buffer.memory, batch_size)
        # Flatten and stack the states for batch processing
        states = np.array([s.flatten() for s, a, r, ns, d in minibatch])
        next_states = np.array([ns.flatten() for s, a, r, ns, d in minibatch])
        actions = torch.tensor([a for s, a, r, ns, d in minibatch], device=self.device)
        rewards = torch.tensor([r for s, a, r, ns, d in minibatch], device=self.device,dtype=torch.float32)
        dones = torch.tensor([d for s, a, r, ns, d in minibatch], device=self.device,dtype=torch.float32)

        # Convert to PyTorch tensors
        states = torch.from_numpy(states).float()
        next_states = torch.from_numpy(next_states).float()
        states = states.to(self.device)
        next_states = next_states.to(self.device)
        
        # Forward pass
        current_q_values = self(states)
        next_q_values = self(next_states).detach()

        # Compute the target
        max_next_q_values = next_q_values.max(1)[0]  # Get max Q value for next states
        target_q_values = current_q_values.clone()
        for idx in range(batch_size):
            target_q_values[idx, actions[idx]] = rewards[idx] + self.gamma * max_next_q_values[idx] * (1 - dones[idx])
            
        # Perform backward pass and optimization
        self.optimizer.zero_grad()  # Clear gradients
        loss = self.criterion(current_q_values, target_q_values)  # Compute loss
        loss.backward()  # Backpropagation
        self.optimizer.step()  # Update weights
        
    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100, epsilon = 0.05,temp = 0.05):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s , info= eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = self.select_action(s, 'greedy', epsilon=epsilon, temp=temp)
                observation, reward, terminated, truncated, info = eval_env.step(a)
                R_ep += reward
                if terminated:
                    break
                else:
                    s = observation
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
