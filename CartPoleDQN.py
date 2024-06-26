import gym
import time
import numpy as np
from Agent import DQNAgent
import sys

# PARAMETERS if not initialized from Experiment.py
num_iterations = 1000 
num_eval_episodes = 10 
eval_interval = 1000  

initial_collect_steps = 100  
collect_steps_per_iteration =   1
replay_buffer_max_length = 10000

batch_size = 64  
log_interval = 200

learning_rate = 1e-3  
gamma = 0.99
epsilon = 0.05
temp = 0.05

def dqn(n_timesteps=num_iterations, use_replay_buffer=True, learning_rate=learning_rate, gamma=gamma, 
        policy="egreedy", epsilon=epsilon, temp=temp, eval_interval=eval_interval, batch_size=batch_size, use_target_network = True):
    
    # Entities
    env = gym.make("CartPole-v1", max_episode_steps=1000)
    env_eval = gym.make("CartPole-v1")
    epsilon_decay = 0.995
    epsilon_min = 0.05
    
    dqn_agent_and_model = DQNAgent(n_states=4, 
                        n_actions=2, 
                        learning_rate=learning_rate, 
                        gamma=gamma,
                        epsilon=epsilon,
                        epsilon_decay=epsilon_decay,
                        epsilon_min=epsilon_min,
                        temp=temp)
                        
    observation, info = env.reset(seed=42) 

    eval_timesteps = []
    eval_returns = []

    iteration = 0
    while iteration <= n_timesteps:
        state, info = env.reset()

        terminated = False
        while not terminated:
            action = dqn_agent_and_model.select_action(state,policy=policy)
            observation, reward, terminated, truncated, info = env.step(action)

            if use_replay_buffer:
                dqn_agent_and_model.remember(state, action, reward, observation, terminated)
                if len(dqn_agent_and_model.replay_buffer) > batch_size:
                    dqn_agent_and_model.replay(batch_size)
            else:
                dqn_agent_and_model.remember(state, action, reward, observation, terminated)
                dqn_agent_and_model.replay(batch_size,use_target_network=use_target_network)        
                dqn_agent_and_model.replay_buffer.clean()
                
            state = observation            
            
            if iteration % eval_interval == 0:
                eval_timesteps.append(iteration)
                eval_returns.append(dqn_agent_and_model.evaluate(env_eval, n_eval_episodes=num_eval_episodes, epsilon = epsilon, temp = temp))
                print("step: ",iteration)

            iteration+=1
            dqn_agent_and_model._current_iteration=iteration

            if iteration >= n_timesteps:
                break
            if terminated:
                break

    dqn_agent_and_model.replay_buffer.clean() 
    env.close()
    
    return np.array(eval_returns), np.array(eval_timesteps) 

