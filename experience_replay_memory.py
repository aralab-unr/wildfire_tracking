# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 18:09:54 2021

@author: kripa
"""

import numpy as np

class ExperienceReplayMemory(object):
    def __init__(self, memory_size, input_dims, n_actions, experience_method):
        super(ExperienceReplayMemory, self).__init__()
        self.max_memory_size = memory_size
        self.counter = 0
        self.experience_method = experience_method    

        self.state_memory = []
        self.next_state_memory = []
        self.reward_memory = []
        self.action_memory = []
        self.done_memory = []
        self.goal_memory = []  
        self.obs_state = []
        self.next_obs_state = []
        
        

    def add_experience(self, state, action, reward, next_state, done, goal, obs_state, next_obs_state):

        curr_index = self.counter % self.max_memory_size
        
        if self.counter < self.max_memory_size:
          self.state_memory.append(state)
          self.action_memory.append(action)
          self.reward_memory.append(reward)
          self.next_state_memory.append(next_state)
          self.done_memory.append(done)
          self.goal_memory.append(goal)
          self.obs_state.append(obs_state)
          self.next_obs_state.append(next_obs_state)
        else:
          self.state_memory[curr_index] = state
          self.action_memory[curr_index] = action
          self.reward_memory[curr_index] = reward
          self.next_state_memory[curr_index] = next_state
          self.done_memory[curr_index] = done
          self.goal_memory[curr_index] = goal
          self.obs_state[curr_index] = obs_state 
          self.next_obs_state[curr_index] = next_obs_state
            
        self.counter += 1
        

    def get_experience(self, batch_size):
        if(self.experience_method == 0):
            rand_state, rand_action, rand_reward, rand_next_state, rand_done, rand_goal, rand_obs_state, rand_next_obs_state = self.get_random_experience(batch_size)
            return rand_state, rand_action, rand_reward, rand_next_state, rand_done, rand_goal, rand_obs_state, rand_next_obs_state
            
            
            
            
    def get_random_experience(self, batch_size):
        sample_size = batch_size
        
        rand_index = np.random.choice(min(self.counter, self.max_memory_size), sample_size, replace=False)
        rand_index = rand_index[0]
        #print("Chose ER Replay Index: ", rand_index)

        rand_state = self.state_memory[rand_index]
        rand_action = self.action_memory[rand_index]
        rand_reward = self.reward_memory[rand_index]
        rand_next_state = self.next_state_memory[rand_index]
        rand_done = self.done_memory[rand_index]
        rand_goal = self.goal_memory[rand_index]
        rand_obs_state = self.obs_state[rand_index]
        rand_next_obs_state = self.next_obs_state[rand_index]

        return rand_state, rand_action, rand_reward, rand_next_state, rand_done, rand_goal, rand_obs_state, rand_next_obs_state
        

# --------------- YHJ ---------------
from collections import deque
import random
import numpy as np
import itertools


class SingleTrajectoryBuffer:
    
    def __init__(self, memory_size):
        self.obs_memory = deque(maxlen=memory_size)
        self.act_memory = deque(maxlen=memory_size)
        self.state_est_memory = deque(maxlen=memory_size)
        self.t = 0
        self.len = 0

    def add(self, obs, act, state_est):
        self.obs_memory.append(obs)
        self.act_memory.append(act)
        self.state_est_memory.append(state_est)
        self.len +=1
    
    def sample(self, n_sample, n_windows):

        batch_obs_stream = []  
        batch_act_stream = []
        batch_state_est_stream = []
        length = len(self.obs_memory)

        for i in range(n_sample):
            start = np.random.choice(length-n_windows)
            stop = min(start + n_windows, length)
            batch_obs_stream.append(list(itertools.islice(self.obs_memory, start, stop)))
            batch_act_stream.append(list(itertools.islice(self.act_memory, start, stop)))
            batch_state_est_stream.append(list(itertools.islice(self.state_est_memory, start, stop)))
            
        return np.array(batch_obs_stream), np.array(batch_act_stream), np.array(batch_state_est_stream)


