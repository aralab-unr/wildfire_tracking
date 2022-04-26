# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 15:01:02 2021

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
from utility_copy import Approx_FSR, FindNextState, FindOptimalJoinAction, RandomJoinAction
from fire_model_test import FireEnvironment
from experience_replay_memory import ExperienceReplayMemory, SingleTrajectoryBuffer
import sys

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from agent import DQN_Agent, DynamicAutoEncoderAgent

def main():
  outputs_dir = "results"
  seed = 0
  np.random.seed(seed)
  random.seed(seed)
  #np.random.seed(seed)
  n_drones = 3
  n_episodes = 1005
  episode_max_steps = 1000
  min_eps = 0.05
  max_eps = 0.95
  eps_decay = 10000
  learning_rate = 0.1
  gamma = 0.9 
  dim_x = 15
  dim_y = 15
  dim_z = 5
  agents_theta_degrees = 30
  
  if os.path.exists(outputs_dir):
    shutil.rmtree(outputs_dir)

  os.makedirs(outputs_dir)


  X_MAX = 14
  X_MIN = 0
  Y_MAX = 14
  Y_MIN = 0
  Z_MAX = 4
  Z_MIN = 0     # z min = 1

  env = FireEnvironment("fbndry4.txt", dim_x, dim_y, [dim_x, dim_y, dim_z], agents_theta_degrees, n_drones, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX)
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

  total_step_run = np.zeros(number_episodes)     
  fov_angle = np.array([30, 30])
  failed_episode = 0    
  #np.random.seed(0)  
  reward_storage = []
  max_reward = 0
  simulation_index = 0
  total_reward = 0
  total_rewards = np.zeros(n_episodes).astype(float)
  simuation_dynamic_fire = 5
  
  replay_memorize_size = 1
  input_dims = 5
  experience_method = 0
  replay = ExperienceReplayMemory(replay_memorize_size, n_drones, 6, experience_method)

  #--- State Estimator ---
  traj_memory_size = 10000
  traj_replay = SingleTrajectoryBuffer(traj_memory_size)
  n_window = 37
  n_sample_traj = 3


  dynenc_agent = DynamicAutoEncoderAgent()

  list_of_dqn_agents = [DQN_Agent(state_dim=9) for i in range(3)]

  prev_state_est = np.random.rand(9)
  
  try:
    for episode in range(n_episodes):
      simulation_index = 0
      total_reward = 0
      np.random.seed(seed)
      random.seed(seed)
      state = env.reset(0)
      max_reward = 0
      print("Reset State: ", state)
      env.map = env.grid
      done = False
      Theta_backup = Theta
      epsilon = epsilon * 0.9
      state_vector  = np.asarray(state)
      terminal_state = 0
      terminal_state_ever = np.zeros(n_episodes)
      local_goal_reached = np.zeros(n_drones)
      for k in range(episode_max_steps):
        if(k % simuation_dynamic_fire == 0):
          simulation_index += 1
          env.grid = env.simStep(simulation_index)
          env.map = env.grid 
          #print(env.map)
          state = env.state()
          state_vector = np.asarray(state)
          #print("~~~~~~~~~~~~Fire Update~~~~~~~~~~~~")
        A = []
        p = np.random.uniform(0,1)



        ##########################################
        ### 1. Get action given state estimate ###
        ##########################################
        curr_action = []
        epsilon_action_det = random.random()
        for i in range(len(list_of_dqn_agents)):
          temp_a = list_of_dqn_agents[i].get_explorative_action(prev_state_est, epsilon_action_det)  
          curr_action.append(temp_a)
          A.append(temp_a)
        
        A = np.array(A)
      
        next_state_vector = np.zeros((n_drones, 3))
        agent_reward_t = np.array([])
        total_reward_t = 0
        total_action_t = 0
        # count step until convergence
        total_step_run[episode] = total_step_run[episode] + 1


        #######################################################
        ### 2. Use the action and get the state observation ###
        #######################################################
        temp_A =  {drone: A[drone] for drone in range(n_drones)}  
        next_state, reward, done, meta = env.step(temp_A)
        next_state_vector = np.asarray(next_state) 
        
        CRi = np.zeros(n_drones)
        total_reward += reward
        CRi.fill(reward)
        if(done):
          CRi.fill(1.0)  

        if(reward > max_reward):
          print("Max Reward: ", reward)
          max_reward = reward


        #####################################
        ### 3. Update the state estimator ###
        #####################################
        observation = next_state_vector
        action = A
        state_est = dynenc_agent.step(observation, action)

        print('state_est', state_est.shape)

        state_est = state_est.squeeze()

        print('state_est', state_est.shape)
        


        

        ### Train the state estimator ###
        replay.add_experience(prev_state_est, A, reward, state_est, done, curr_action)
        traj_replay.add(state, A, state_est)

        print(traj_replay.len)


        prev_state_est = state_est


        if traj_replay.len > n_window:


          '''
          #################
          ### DQN Train ###
          #################
          for i in range(len(list_of_dqn_agents)):
            minibatch_actions_for_this_agent = minibatch_actions_for_agents.T[i]
            list_of_dqn_agents[i].update(minibatch_state, minibatch_actions_for_this_agent, minibatch_reward, minibatch_next_state, minibatch_done)

          #################################
          ### Dynamic AutoEncoder Train ###
          #################################
          batch_obs_stream, batch_act_stream, batch_state_est_stream = traj_replay.sample(n_sample_traj, n_window)
          loss_val = dynenc_agent.update(batch_obs_stream, batch_state_est_stream, batch_act_stream)
          '''


        # -------------------------------


        n_minibatch = 1
        minibatch_state = []
        minibatch_actions_for_agents = []
        minibatch_next_state = []
        minibatch_reward = []
        minibatch_done = []
        
        for i in range(n_minibatch):
          state_1, A_1, reward_1, next_state_1, done_1, A_next_1 = replay.get_random_experience(1)  # It looks the random sampling is not working!
          minibatch_state.append(np.array(state_1).flatten())
          minibatch_actions_for_agents.append(np.array(A_1))
          minibatch_next_state.append(np.array(next_state_1).flatten())
          minibatch_reward.append(reward_1)
          minibatch_done.append(done)
          
        minibatch_state = np.array(minibatch_state)
        minibatch_actions_for_agents = np.array(minibatch_actions_for_agents)
        minibatch_next_state = np.array(minibatch_next_state)
        minibatch_reward = np.array(minibatch_reward)
        minibatch_done = np.array(minibatch_done)

        print('minibatch_state for DQN', minibatch_state.shape)
        
#        for i in range(len(list_of_dqn_agents)):
#          minibatch_actions_for_this_agent = minibatch_actions_for_agents.T[i]
#          loss_val = list_of_dqn_agents[i].update(minibatch_state, minibatch_actions_for_this_agent, minibatch_reward, minibatch_next_state, minibatch_done)
          
        if traj_replay.len > n_window:
          #################
          ### DQN Train ###
          #################
          for i in range(len(list_of_dqn_agents)):
            minibatch_actions_for_this_agent = minibatch_actions_for_agents.T[i]
            list_of_dqn_agents[i].update(minibatch_state, minibatch_actions_for_this_agent, minibatch_reward, minibatch_next_state, minibatch_done)

          #################################
          ### Dynamic AutoEncoder Train ###
          #################################
          batch_obs_stream, batch_act_stream, batch_state_est_stream = traj_replay.sample(n_sample_traj, n_window)
          loss_val = dynenc_agent.update(batch_obs_stream, batch_state_est_stream, batch_act_stream)          
        
          
        episode_rewards[episode] += reward
        reward_storage.append(reward)
        state = next_state
        state_vector = next_state_vector
        episode_steps[episode] += 1
        total_rewards[episode] += reward
        steps += 1
        
      print("Total Reward: ", total_reward)  
      print(f'Episode {episode}: {k + 1} steps.')
  except KeyboardInterrupt:
      pass
    
  #print(total_rewards)
  plt.plot(total_rewards)
  plt.savefig(os.path.join(outputs_dir, 'episode_steps.png'))
  pickle.dump(episode_steps, open(os.path.join(outputs_dir, 'episode_steps.pkl'), 'wb'))  
    
if __name__ == '__main__':
    main()