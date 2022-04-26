import argparse
import numpy as np
from field_coverage_env import FieldCoverageEnv
from itertools import product, combinations
from cvxopt import solvers, matrix
from copy import deepcopy
import math
import matplotlib.pyplot as plt
import pickle
import shutil
import os
import time
import random
from random import randint
import copy

optimal_counter = 0
optimal_joint_action_global = []

def Approx_FSR(agent_i, current_state_agent_i, join_action, n_drones, action_space, env_shape):
    '''
    X, Y, Z = env_shape
    state_dim = (X + Y + Z) * n_drones
    drone_actions = np.arange(action_space)
    actions = {x: i for i, x in enumerate(product(*((drone_actions,) * n_drones)))}
    states = []
    x = current_state_agent_i[0]
    y = current_state_agent_i[1]
    z = current_state_agent_i[2]

    arr = np.zeros(X)
    arr[x] = 1
    states.append(arr)

    arr = np.zeros(Y)
    arr[y] = 1
    states.append(arr)

    arr = np.zeros(Z)
    arr[z] = 1
    states.append(arr)
    states = np.concatenate(states)
    state = np.zeros(len(states) * (action_space ** n_drones))
    action_slot = join_action
    state[action_slot: action_slot + len(states)] = state
    '''
    X, Y, Z = env_shape
    length = X + Y + Z 
    phi_BF = np.zeros((length * n_drones * (action_space ** n_drones), 1))
    agent_i_x = current_state_agent_i[0]
    agent_i_y = current_state_agent_i[1]
    agent_i_z = current_state_agent_i[2]
    phi_BF[length*(agent_i - 1) + agent_i_x + (join_action-1)*length*n_drones, 0] = 1
    phi_BF[length*(agent_i - 1) + X + agent_i_y + (join_action-1)*length*n_drones, 0] = 1
    phi_BF[length*(agent_i - 1) + X + Y + agent_i_z + (join_action-1)*length*n_drones, 0] = 1
    return phi_BF

def FindNextState(current_state_agent_i, next_action, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, number_drones, number_actions, env_shape):
    agent_i_x = current_state_agent_i[0]
    agent_i_y = current_state_agent_i[1]
    agent_i_z = current_state_agent_i[2]

    if next_action == 1:
        if agent_i_x - 1 >= X_MIN:
            agent_i_x_new = agent_i_x - 1
            next_state = np.array([agent_i_x_new, agent_i_y, agent_i_z])
        else:
            agent_i_x_new = X_MIN
            next_state = np.array([agent_i_x_new, agent_i_y, agent_i_z])
    elif next_action == 2:
        if agent_i_x + 1 <= X_MAX:
            agent_i_x_new = agent_i_x + 1
            next_state = np.array([agent_i_x_new, agent_i_y, agent_i_z])
        else:
            agent_i_x_new = X_MAX
            next_state = np.array([agent_i_x_new, agent_i_y, agent_i_z])
    elif next_action == 3:
        if agent_i_y + 1 <= Y_MAX:
            agent_i_y_new = agent_i_y + 1
            next_state = np.array([agent_i_x, agent_i_y_new, agent_i_z])
        else:
            agent_i_y_new = Y_MAX
            next_state = np.array([agent_i_x, agent_i_y_new, agent_i_z])
    elif next_action == 4:
        if agent_i_y - 1 >= Y_MIN:
            agent_i_y_new = agent_i_y - 1
            next_state = np.array([agent_i_x, agent_i_y_new, agent_i_z])
        else:
            agent_i_y_new = Y_MIN;
            next_state = np.array([agent_i_x, agent_i_y_new, agent_i_z])
    elif next_action == 5:
        if agent_i_z + 1 <= Z_MAX:
            agent_i_z_new = agent_i_z + 1
            next_state = np.array([agent_i_x, agent_i_y, agent_i_z_new])
        else:
            agent_i_z_new = Z_MAX
            next_state = np.array([agent_i_x, agent_i_y, agent_i_z_new])
    elif next_action == 6:
        if agent_i_z - 1 >= Z_MIN:
            agent_i_z_new = agent_i_z - 1
            next_state = np.array([agent_i_x, agent_i_y, agent_i_z_new])
        else:
            agent_i_z_new = Z_MIN
            next_state = np.array([agent_i_x, agent_i_y, agent_i_z_new])
    else:
        next_state = np.array([agent_i_x, agent_i_y, agent_i_z])

    return next_state



def FindOptimalJoinActionHelper(i, all_actions, action_vector, V, number_actions, Theta, F, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, number_drones, num_actions, env_shape, current_agent_states, t_next):
  global optimal_counter
  global optimal_joint_action_global
  for j in range(len(all_actions[i])):
    t_vector = copy.copy(action_vector)
    next_state = FindNextState(current_agent_states[i], all_actions[i][j], X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, 
              number_drones, num_actions, env_shape).astype(int)
    if(F[next_state[0], next_state[1]] > 0) and (not np.array_equal(t_next, next_state)) and (not np.array_equal(next_state, current_agent_states[i])) and (i < (len(all_actions)) - 1):
      t_vector.append(all_actions[i][j])
      FindOptimalJoinActionHelper(i + 1, all_actions, t_vector, V, number_actions, Theta, F, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, 
                            number_drones, num_actions, env_shape, current_agent_states, next_state)
    elif(F[next_state[0], next_state[1]] > 0) and (not np.array_equal(t_next, next_state)) and (not np.array_equal(next_state, current_agent_states[i])) and (i == (len(all_actions)) - 1):
      t_vector.append(all_actions[i][j])
      optimal_joint_action_global.append(t_vector)
      optimal_counter += 1



def FindOptimalJoinAction(V, number_actions, Theta, F, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, number_drones, num_actions, env_shape):
    global optimal_counter
    global optimal_joint_action_global

    
    current_agent_states = []
    for i in range(len(V)):
      current_agent_states.append(V[i])

    current_agent_states = np.asarray(current_agent_states)


    num_rows, num_cols = V.shape
    Q_A = np.array([])
    # Find possible value
    counter = 0
    all_actions = []
    for i in range(number_drones):
      t_actions = []
      for j in range(1, number_actions + 1):
        t_actions.append(j)
      all_actions.append(t_actions)
    
    i = 0 
    t_counter = 0
    #print("prior: ", optimal_counter)
    for j in range(len(all_actions[i])):
      action_vector = []
      next_state = FindNextState(current_agent_states[i], all_actions[i][j], X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, 
                    number_drones, num_actions, env_shape).astype(int)
      if(F[next_state[0], next_state[1]] > 0) and (not np.array_equal(next_state, current_agent_states[i])):
        action_vector.append(all_actions[i][j])
        FindOptimalJoinActionHelper(i + 1, all_actions, action_vector, V, number_actions, Theta, F, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, 
                                    number_drones, num_actions, env_shape, current_agent_states, next_state)

    #print("Optimal Global: ", optimal_counter, len(optimal_joint_action_global))
    optimal_counter = 0
    

    for x in range(len(optimal_joint_action_global)):
      #print(optimal_joint_action_global[x])
      join_action = 0
      phi_bf = []
      q_i = []
      ce = 1
      for y in range(len(optimal_joint_action_global[x])):
        if(y < (len(optimal_joint_action_global[x]) - 1)):
          join_action += (optimal_joint_action_global[x][y] - 1) * (number_actions ** (number_drones - (y + 1)))
        else:
          join_action += (optimal_joint_action_global[x][y] - 1)
      for z in range(len(optimal_joint_action_global[x])):
        phi_bf.append(Approx_FSR(z + 1, current_agent_states[z], join_action, number_drones, number_actions, env_shape))
        q_i.append(np.round(np.matmul(Theta[:, z], phi_bf[z])[0], 3))
        ce *= q_i[z]
      Q_A_i =  np.zeros(number_drones + 2)
      for w in range(number_drones):
        Q_A_i[w] = optimal_joint_action_global[x][w]
      #np.array([possible_action_1, possible_action_2, possible_action_3, join_action, CE])
      Q_A_i[number_drones] = join_action
      Q_A_i[number_drones + 1] = ce
      if Q_A.size == 0:
          Q_A = Q_A_i
      else:
          Q_A = np.vstack((Q_A, Q_A_i))

    optimal_joint_action_global.clear()
    #print("Optimal: ", counter)
    # Determine the maximum one
    if Q_A.size != 0:
        #print("Optimal joint action solution found!")
        if Q_A[0].size > 1:
            Q_A_CE_array = Q_A[:, number_drones + 1]

            index_optimal_array = np.array([])
            for i in range(0, Q_A_CE_array.size):
                if np.isclose(Q_A_CE_array[i], Q_A_CE_array[np.argmax(Q_A_CE_array)]):
                    if index_optimal_array.size == 0:
                        index_optimal_array = np.array(i)
                    else:
                        index_optimal_array = np.hstack((index_optimal_array, np.array(i)))
            # Determine returning
            if index_optimal_array.size > 1:    # More than one optimal value
                random_optimal_index = randint(0, index_optimal_array.size - 1)   # Take one index randomly
                action_set = Q_A[index_optimal_array[random_optimal_index], 0:number_drones]    # Meaning take value from index 0, 1, 2 (so up to 3)
                # print "optimal action found! (multi)"
                # print Q_A_CE_array[random_optimal_index], max(Q_A_CE_array), min(Q_A_CE_array)

                return action_set
            elif index_optimal_array.size == 1: # one unique optimal value
                action_set = Q_A[index_optimal_array, 0:number_drones]  # meaning take value from index 0, 1, 2 (so up to 3)
                # print "optimal action found! (one)"
                # print "unique CE", Q_A_CE_array
                return action_set
            else:   # none optimal value
                random_optimal_index = randint(0, Q_A_CE_array.size - 1)
                action_set = Q_A[index_optimal_array[random_optimal_index], 0:number_drones]  # meaning take value from index 0, 1, 2 (so up to 3)
                return action_set
        else:
            action_set = Q_A[0:number_drones]
            return action_set
    else:
        action_set = Q_A
        '''
        action_set = []
        for drone in range(number_drones):
          action_set.append(np.random.choice(number_actions) + 1)
        action_set = np.array(action_set)
        '''
        print ("V=", current_state_agent_1, current_state_agent_2, current_state_agent_3)     # flags if empty array
        return action_set


def RandomJoinActionHelper(i, all_actions, action_vector, V, number_actions, F, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, number_drones, num_actions, env_shape, current_agent_states, t_next):
  global optimal_counter
  global optimal_joint_action_global
  for j in range(len(all_actions[i])):
    t_vector = copy.copy(action_vector)
    next_state = FindNextState(current_agent_states[i], all_actions[i][j], X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, 
              number_drones, num_actions, env_shape).astype(int)
    if(F[next_state[0], next_state[1]] > 0) and (not np.array_equal(t_next, next_state)) and (not np.array_equal(next_state, current_agent_states[i])) and (i < (len(all_actions)) - 1):
      t_vector.append(all_actions[i][j])
      RandomJoinActionHelper(i + 1, all_actions, t_vector, V, number_actions, F, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, 
                            number_drones, num_actions, env_shape, current_agent_states, next_state)
    elif(F[next_state[0], next_state[1]] > 0) and (not np.array_equal(t_next, next_state)) and (not np.array_equal(next_state, current_agent_states[i])) and (i == (len(all_actions)) - 1):
      t_vector.append(all_actions[i][j])
      optimal_joint_action_global.append(t_vector)
      optimal_counter += 1


def RandomJoinAction(V, number_actions, F, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, number_drones, num_actions, env_shape):
    global optimal_counter
    global optimal_joint_action_global

    
    current_agent_states = []
    for i in range(len(V)):
      current_agent_states.append(V[i])

    current_agent_states = np.asarray(current_agent_states)


    num_rows, num_cols = V.shape
    Q_A = np.array([])
    # Find possible value
    counter = 0
    all_actions = []
    for i in range(number_drones):
      t_actions = []
      for j in range(1, number_actions + 1):
        t_actions.append(j)
      all_actions.append(t_actions)
    
    i = 0 
    t_counter = 0
    #print("prior: ", optimal_counter)
    for j in range(len(all_actions[i])):
      action_vector = []
      next_state = FindNextState(current_agent_states[i], all_actions[i][j], X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, 
                    number_drones, num_actions, env_shape).astype(int)
      if(F[next_state[0], next_state[1]] > 0) and (not np.array_equal(next_state, current_agent_states[i])):
        action_vector.append(all_actions[i][j])
        RandomJoinActionHelper(i + 1, all_actions, action_vector, V, number_actions, F, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, 
                                    number_drones, num_actions, env_shape, current_agent_states, next_state)

    #print("Optimal Global: ", optimal_counter, len(optimal_joint_action_global))
    optimal_counter = 0

    for x in range(len(optimal_joint_action_global)):
      #print(optimal_joint_action_global[x])
      join_action = 0
      phi_bf = []
      q_i = []
      ce = 1
      Q_A_i =  np.zeros(number_drones)
      for y in range(len(optimal_joint_action_global[x])):
        Q_A_i[y] = optimal_joint_action_global[x][y]
      #np.array([possible_action_1, possible_action_2, possible_action_3])
      if Q_A.size == 0:
          Q_A = Q_A_i
      else:
          Q_A = np.vstack((Q_A, Q_A_i))

    optimal_joint_action_global.clear()
    
    #print("Random: ", Q_A.size)
    # return a random action
    if Q_A.size != 0:
        if (Q_A[0].size > 1):   #check if an array is not 1D
            num_rows, num_cols = Q_A.shape
            random_index = np.random.randint(0, num_rows - 1)  # take one index randomly
            action_set = Q_A[random_index, 0:number_drones]  # meaning take value from index 0, 1, 2 (so up to 3)
        else:
            action_set = Q_A
    else:
        action_set = Q_A
        '''
        action_set = []
        for drone in range(number_drones):
          action_set.append(np.random.choice(number_actions) + 1)
        action_set = np.array(action_set)
        '''
        #print(action_set)
        print ("V=", current_state_agent_1, current_state_agent_2, current_state_agent_3)     # flags if empty array
    return action_set

