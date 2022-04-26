# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 20:47:45 2021

@author: kripa
"""

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
from utility_copy import Approx_FSR, FindNextState, FindOptimalJoinAction, RandomJoinAction
from fire_model_test import FireEnvironment
from experience_replay_memory import ExperienceReplayMemory
import sys



def main():
  outputs_dir = "results"
  seed = 0
  np.random.seed(seed)
  n_drones = 3
  n_episodes = 700
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


  np.random.seed(seed)
  random.seed(seed)

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


  #----------YHJ--------------#
  from nn_networks import DQN
  import torch, random
  DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  epsilon_greedy = 0.2


  class DQN_Agent:

    def __init__(self):
      self.dqn_network = DQN(state_dim=9, n_action=6).to(DEVICE)
      self.target_dqn_network = DQN(state_dim=9, n_action=6).to(DEVICE)
      self.nn_optimizer = torch.optim.Adam(self.dqn_network.parameters())
      self.loss_criterion = torch.nn.HuberLoss()

      self._iteration = 0

    def update(self, batch_state, batch_actions, batch_reward, batch_next_state, batch_done):
      # Copied from https://github.com/transedward/pytorch-dqn/blob/master/dqn_learn.py

      # Compute current Q value, q_func takes only state and output value for every state-action pair
      # We choose Q based on action taken.
      torch_state_batch = torch.FloatTensor(np.array(batch_state)).to(DEVICE)
      torch_action_batch = torch.LongTensor(np.array(batch_actions)).to(DEVICE).unsqueeze(-1)
      torch_action_batch = torch_action_batch - 1  # [1, 6] -> [0, 5]
      current_Q_values = self.dqn_network(torch_state_batch).gather(1, torch_action_batch)  # Q(s,a)

      # Compute next Q value based on which action gives max Q values
      # Detach variable from the current graph since we don't want gradients for next Q to propagated
      torch_next_state_batch = torch.FloatTensor(np.array(batch_next_state)).to(DEVICE)
      next_max_q = self.target_dqn_network(torch_next_state_batch).detach().max(1)[0]
      #next_Q_values = not_done_mask * next_max_q <--- We are not doing this yet.
      next_Q_values = next_max_q

      # Compute the target of the current Q values
      torch_reward_batch = torch.FloatTensor(np.array(batch_reward)).to(DEVICE)
      target_Q_values = torch_reward_batch + (discount_rate * next_Q_values)

      # Compute Bellman error
      loss = self.loss_criterion(current_Q_values.squeeze(), target_Q_values.squeeze())

      self.nn_optimizer.zero_grad()
      loss.backward()
      self.nn_optimizer.step()

      loss_val = loss.item()

      self._iteration += 1

      if self._iteration%10 == 0:
        self.target_dqn_network.load_state_dict(self.dqn_network.state_dict())
        print('target nn update') 
      return loss_val
    
    def get_exploit_action(self, state):
      torch_state = torch.FloatTensor(np.array(state)).to(DEVICE)
      q_for_all_action = self.target_dqn_network(torch_state).detach()
      action = torch.argmax(q_for_all_action)
      return action.item()+1 # because he uses [1, 6] instead of [0,5]

    def get_explorative_action(self, state):
      if random.random() > epsilon_greedy:
        action = self.get_exploit_action(state)
      else:
        action = random.randit(0,5)
      return action # because he uses [1, 6] instead of [0,5]





      

  
  list_of_dqn_agents = [DQN_Agent() for i in range(3)]




        
        
        
        
        



  
  try:
    for episode in range(n_episodes):
      simulation_index = 0
      total_reward = 0
      state = env.reset(0)
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
        if(k % simuation_dynamic_fire == 0):
          simulation_index += 1
          env.grid = env.simStep(simulation_index)
          env.map = env.grid 
          #print(env.map)
          state = env.state()
          V = np.asarray(state)
          #print("~~~~~~~~~~~~Fire Update~~~~~~~~~~~~")
          
        A = None
        p = np.random.uniform(0,1)
        
        #print("V: ", env.map[V[0][0], V[0][1]], env.map[V[1][0], V[1][1]], env.map[V[1][0], V[1][1]])

        if(p >= epsilon):
          A = FindOptimalJoinAction(V, action_space, Theta, env.map, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, n_drones, action_space, env_dim)
        else:
          A = RandomJoinAction(V, action_space, env.map, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, n_drones, action_space, env_dim)
        
        if A.size == 0:         # Prevent empty case
          terminal_state = True
          failed_episode = failed_episode + 1
          Theta = Theta_backup
          print ("Failed episode! Attempt to redo the episode")
          sys.exit(0)
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
          total_reward += reward
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
          
          before_er_next_state = next_state
          before_er_Vnew = Vnew
          before_er_reward = reward
          if(len(A_next) > 0):
            replay.add_experience(state, A, reward, next_state, done, A_next)
          state, A, reward, next_state, done, A_next = replay.get_experience(1)  # It looks that I cannot get more than one sample at a time.

          #----------YHJ--------------#

          n_minibatch = 5
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
          

          for i in range(len(list_of_dqn_agents)):
            minibatch_actions_for_this_agent = minibatch_actions_for_agents.T[i]
            list_of_dqn_agents[i].update(minibatch_state, minibatch_actions_for_this_agent, minibatch_reward, minibatch_next_state, minibatch_done)

          
          for i in range(len(list_of_dqn_agents)):
            list_of_dqn_agents[i].get_exploit_action(np.array(state).flatten())






          #break






          
          V = np.asarray(state)
          Vnew = np.asarray(next_state)
          
          '''
          print("~~~~~~~~~~~~~~~~")
          print("ER state: ", state)
          print("ER Actions: ", A)                    
          print("ER Next State: ", next_state)
          print("ER Next Action: ", A_next)
          print("~~~~~~~~~~~~~~~~")
          '''

          #print("state: ", env.map[state[0][0], state[0][1]], env.map[state[1][0], state[1][1]], env.map[state[1][0], state[1][1]])
          #print("next state: ", env.map[next_state[0][0], next_state[0][1]], env.map[next_state[1][0], next_state[1][1]], env.map[next_state[1][0], next_state[1][1]])
          
          action_taken = int((A[0] - 1) * (action_space ** (n_drones - 1)) + (A[1] - 1) * (action_space ** (n_drones - 2)) + A[2])
          opt_join_action = int((A_next[0] - 1) * (action_space ** (n_drones - 1)) + (A_next[1] - 1) * (action_space ** (n_drones - 2)) + A_next[2])        
          

          for agent_i in range(0, n_drones):
              previous_state_agent_i = V[agent_i] 
              new_state_agent_i = Vnew[agent_i]
              phi_BF_previous = Approx_FSR((agent_i + 1), previous_state_agent_i, action_taken, n_drones, action_space, env_dim)
              phi_BF_new = Approx_FSR((agent_i + 1), new_state_agent_i, opt_join_action, n_drones, action_space, env_dim)
              TD_diff_i = learning_rate * (CRi[agent_i] + np.round(discount_rate * np.matmul(Theta[:, agent_i], phi_BF_new)[0] - np.matmul(Theta[:, agent_i], phi_BF_previous)[0], 5))*phi_BF_previous
              Theta[:, agent_i:agent_i+1] = Theta[:, agent_i:agent_i+1] + TD_diff_i 

          episode_rewards[episode] += before_er_reward
          reward_storage.append(before_er_reward)
          state = before_er_next_state
          V = before_er_Vnew
          #print("time: ", time.time() - t0)
          #print("episode: ", episode, " steps: ", k)
          #if done:
            #("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Done!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
          #  break
          episode_steps[episode] += 1
          total_rewards[episode] += before_er_reward
          steps += 1

      break
      #print(f'Episode {episode}: {k + 1} steps.')
      #print("Total Reward: ", total_reward)
  except KeyboardInterrupt:
      pass
  #plt.ylim(0, episode_max_steps)
  #plt.plot(episode_steps)
  # print(total_rewards)
  # plt.plot(total_rewards)
  # plt.savefig(os.path.join(outputs_dir, 'episode_steps.png'))
  # pickle.dump(episode_steps, open(os.path.join(outputs_dir, 'episode_steps.pkl'), 'wb'))  
  
  
  
  
if __name__ == '__main__':
    main()