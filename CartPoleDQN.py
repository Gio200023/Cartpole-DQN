import gym
import time
import numpy as np
from Agent import DQNAgent
import sys

# PARAMETERS 
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

def dqn(n_timesteps=num_iterations, learning_rate=learning_rate, gamma=gamma, policy="egreedy", epsilon=epsilon, temp=temp, eval_interval=eval_interval):
    # Entities
    env = gym.make("CartPole-v1", render_mode="", max_episode_steps=1000)
    env_eval = gym.make("CartPole-v1")

    dqn_agent_and_model = DQNAgent(n_states=env.observation_space.shape[0], 
                        n_actions=env.action_space.n, 
                        learning_rate=learning_rate, 
                        gamma=gamma,
                        epsilon=epsilon,
                        temp=temp)
    observation, info = env.reset(seed=42) 

    eval_timesteps = []
    eval_returns = []

    iteration = 0
    while iteration <= n_timesteps:
        state, info = env.reset()
        # state = np.reshape(state, [1, dqn_agent_and_model.n_states])
        terminated = False
        
        while not terminated:
            action = dqn_agent_and_model.select_action(state,epsilon=epsilon, temp=temp)
            observation, reward, terminated, truncated, info = env.step(action)
            observation = np.reshape(observation, [1, dqn_agent_and_model.n_states])
            dqn_agent_and_model.remember(state, action, reward, observation, terminated)

            if len(dqn_agent_and_model.replay_buffer) >= batch_size:
                dqn_agent_and_model.replay(batch_size)
            state = observation            
            if iteration % eval_interval == 0:
                eval_timesteps.append(iteration)
                eval_returns.append(dqn_agent_and_model.evaluate(env_eval, n_eval_episodes=num_eval_episodes, epsilon = epsilon, temp = temp))
                print(f"Average return at iteration {iteration}: {eval_returns}")
            iteration+=1
            
            if terminated:
                break
            
                
    env.close()
    
    return np.array(eval_returns), np.array(eval_timesteps) 

