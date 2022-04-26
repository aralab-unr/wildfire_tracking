# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 17:08:33 2021

@author: kripa
"""

import argparse
import numpy as np
from field_coverage_env import FieldCoverageEnv
from copy import deepcopy
import math
import matplotlib.pyplot as plt
import pickle
import shutil
import os
import time
import random
from random import randint
from utility import Approx_FSR, FindNextState, FindOptimalJoinAction, RandomJoinAction
from fire_model_test import FireEnvironment



def main():
  outputs_dir = "results"
  seed = 0
  np.random.seed(seed)
  n_drones = 4
  n_episodes = 700
  episode_max_steps = 2000
  min_eps = 0.05
  max_eps = 0.95
  eps_decay = 10000
  learning_rate = 0.1
  gamma = 0.9 
  dim_x = 7
  dim_y = 7
  dim_z = 3
  agents_theta_degrees = 30
  
  if os.path.exists(outputs_dir):
    shutil.rmtree(outputs_dir)

  os.makedirs(outputs_dir)


  np.random.seed(seed)
  random.seed(seed)

  env = FireEnvironment("fbndry4.txt", dim_x, dim_y, [dim_x, dim_y, dim_z], agents_theta_degrees, n_drones)
  env_dim = (dim_x, dim_y, dim_z)
  X, Y, Z = env_dim
  state_dim = (X + Y + Z)
  action_space = 6
  Theta = np.zeros(((X + Y + Z) * n_drones * (action_space ** n_drones), n_drones))
  episode_rewards = np.zeros(n_episodes)
  episode_steps = np.zeros(n_episodes).astype(int)
  number_episodes = n_episodes
  num_time_steps = episode_max_steps
  number_time_steps = num_time_steps
  steps = 0
  epsilon = 1
  learning_rate = 0.1
  discount_rate = 0.9
  learning_episode_run = np.linspace(1, number_episodes*number_time_steps, num=number_episodes*number_time_steps)
  X_MAX = 6
  X_MIN = 0
  Y_MAX = 6
  Y_MIN = 0
  Z_MAX = 3
  Z_MIN = 1     # z min = 1
  total_step_run = np.zeros(number_episodes)     
  fov_angle = np.array([30, 30])
  failed_episode = 0    
  #np.random.seed(0)  
  reward_storage = []
  max_reward = 0
  
  try:
    for episode in range(n_episodes):
      state = env.reset()
      max_reward = 0
      #print("Reset State: ", state)
      env.map = env.grid
      done = False
      Theta_backup = Theta
      epsilon = epsilon * 0.9
      V = np.asarray(state)
      terminal_state = 0
      terminal_state_ever = np.zeros(n_episodes)
      local_goal_reached = np.zeros(n_drones)
      for k in range(episode_max_steps):
        #t0 = time.time()
        A = None
        p = np.random.uniform(0,1)
        
        if(p >= epsilon):
          A = FindOptimalJoinAction(V, action_space, Theta, env.map, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, n_drones, action_space, env_dim)
        else:
          A = RandomJoinAction(V, action_space, env.map, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, n_drones, action_space, env_dim)
        
        if A.size == 0:         # Prevent empty case
          terminal_state = True
          failed_episode = failed_episode + 1
          Theta = Theta_backup
          print ("Failed episode! Attempt to redo the episode")
          if(episode == 0):
            episode = 0
          else:
            episode = episode - 1
        else:
          Vnew = np.zeros((n_drones, 3))
          agent_reward_t = np.array([])
          total_reward_t = 0
          total_action_t = 0
          # count step until convergence
          total_step_run[episode] = total_step_run[episode] + 1

          #print("Actions: " , A)
          temp_A =  {drone: A[drone] for drone in range(n_drones)}
          #print("Actions mod: ", A)

          next_state, reward, done, meta = env.step(temp_A)
          #env.render()
          Vnew = np.asarray(next_state) 
          #CRi = np.array([0, 0, 0], dtype=np.float64)
          CRi = np.zeros(n_drones)
          CRi.fill(reward)
          if(done):
            CRi.fill(1.0)
                  

          A_next = FindOptimalJoinAction(Vnew, action_space, Theta, env.map, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, n_drones, action_space, env_dim)
          #print("reward: ", reward)
          
          '''
          print("~~~~~~~~~~~~~~~~")
          print("state: ", state)
          print("Actions: ", A)                    
          print("Next State: ", next_state)
          print("Next Action: ", A_next)
          print("~~~~~~~~~~~~~~~~")
          '''
          if(reward > max_reward):
            print("Max Reward: ", reward)
            max_reward = reward
          

          action_taken = int((A[0] - 1) * (action_space ** (n_drones - 1)) + (A[1] - 1) * (action_space ** (n_drones - 2)) + (A[2] - 1))
          opt_join_action = int((A_next[0] - 1) * (action_space ** (n_drones - 1)) + (A_next[1] - 1) * (action_space ** (n_drones - 2)) + (A_next[2] - 1))          
  
          for agent_i in range(0, n_drones):
              previous_state_agent_i = V[agent_i] 
              new_state_agent_i = Vnew[agent_i]
              phi_BF_previous = Approx_FSR((agent_i + 1), previous_state_agent_i, action_taken, n_drones, action_space, env_dim)
              phi_BF_new = Approx_FSR((agent_i + 1), new_state_agent_i, opt_join_action, n_drones, action_space, env_dim)
              TD_diff_i = learning_rate * (CRi[agent_i] + np.round(discount_rate * np.matmul(Theta[:, agent_i], phi_BF_new)[0] - np.matmul(Theta[:, agent_i], phi_BF_previous)[0], 5))*phi_BF_previous
              Theta[:, agent_i:agent_i+1] = Theta[:, agent_i:agent_i+1] + TD_diff_i 

          episode_rewards[episode] += reward
          reward_storage.append(reward)
          state = next_state
          V = Vnew
          #print("time: ", time.time() - t0)
          #print("episode: ", episode, " steps: ", k)
          if done:
            #("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Done!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            break
          episode_steps[episode] += 1
          steps += 1
      print(f'Episode {episode}: {k + 1} steps.')
  except KeyboardInterrupt:
      pass
  #plt.ylim(0, episode_max_steps)
  plt.plot(episode_steps)
  plt.savefig(os.path.join(outputs_dir, 'episode_steps.png'))
  pickle.dump(episode_steps, open(os.path.join(outputs_dir, 'episode_steps.pkl'), 'wb'))  
  
  
  
  
if __name__ == '__main__':
    main()