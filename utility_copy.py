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
    #print(current_state_agent_i, join_action)
    phi_BF[length*(agent_i - 1) + agent_i_x + (join_action-1)*length*n_drones, 0] = 1
    phi_BF[length*(agent_i - 1) + X + agent_i_y + (join_action-1)*length*n_drones, 0] = 1
    phi_BF[length*(agent_i - 1) + X + Y + agent_i_z + (join_action-1)*length*n_drones, 0] = 1
    return phi_BF

def FindNextState(current_state_agent_i, next_action, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, number_drones, number_actions, env_shape):
    agent_i_x = current_state_agent_i[0]
    agent_i_y = current_state_agent_i[1]
    agent_i_z = current_state_agent_i[2]

    if next_action == 1:
        if agent_i_x + 1 <= X_MAX:
            agent_i_x_new = agent_i_x + 1
            next_state = np.array([agent_i_x_new, agent_i_y, agent_i_z])
        else:
            agent_i_x_new = X_MAX
            next_state = np.array([agent_i_x_new, agent_i_y, agent_i_z])
    elif next_action == 2:
        if agent_i_x - 1 >= X_MIN:
            agent_i_x_new = agent_i_x - 1
            next_state = np.array([agent_i_x_new, agent_i_y, agent_i_z])
        else:
            agent_i_x_new = X_MIN
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
            agent_i_y_new = Y_MIN
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


def FindOptimalJoinAction(V, number_actions, Theta, F, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, number_drones, num_actions, env_shape):
    current_state_agent_1 = V[0]
    current_state_agent_2 = V[1]
    current_state_agent_3 = V[2]
    
    current_agent_states = []
    for i in range(len(V)):
      current_agent_states.append(V[i])

    current_agent_states = np.asarray(current_agent_states)


    num_rows, num_cols = V.shape
    number_drones = num_rows
    Q_A = np.array([])
    # Find possible value
    counter = 0

    for possible_action_1 in range(1, number_actions+1):
        next_state_agent_1 = FindNextState(current_state_agent_1, possible_action_1, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, number_drones, num_actions, env_shape).astype(int)
        #print("1: " , possible_action_1)
        #print(F[next_state_agent_1[0], next_state_agent_1[1]] > 0)
        if (F[next_state_agent_1[0], next_state_agent_1[1]] > 0) and (not np.array_equal(next_state_agent_1, current_state_agent_1)):   # Evaluate selectable actions to eliminate inappropriate actions
            for possible_action_2 in range(1, number_actions+1):
                next_state_agent_2 = FindNextState(current_state_agent_2, possible_action_2, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, number_drones, num_actions, env_shape).astype(int)
                #print("2 : ", F[next_state_agent_2[0], next_state_agent_2[1]] > 0)
                if (F[next_state_agent_2[0], next_state_agent_2[1]] > 0) and (not np.array_equal(next_state_agent_1, next_state_agent_2)) and (not np.array_equal(next_state_agent_2, current_state_agent_2)):
                    for possible_action_3 in range(1, number_actions+1):
                        next_state_agent_3 = FindNextState(current_state_agent_3, possible_action_3, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, number_drones, num_actions, env_shape).astype(int)
                        #print("3: ", F[next_state_agent_3[0], next_state_agent_3[1]] > 0)
                        if (F[next_state_agent_3[0], next_state_agent_3[1]] > 0) and (not np.array_equal(next_state_agent_1, next_state_agent_3)) and (not np.array_equal(next_state_agent_2, next_state_agent_3)) and (not np.array_equal(next_state_agent_3, current_state_agent_3)):
                            #print(next_state_agent_1, next_state_agent_2, next_state_agent_3)
                            #print(current_state_agent_3, next_state_agent_3)
                            #print(possible_action_1, possible_action_2, possible_action_3)
                            counter = counter + 1
                            #print(counter)
                            join_action = (possible_action_1 - 1) * (number_actions**(number_drones - 1)) + (possible_action_2 - 1) * (number_actions**(number_drones - 2)) + possible_action_3
                            #print(join_action)
                            phi_BF1 = Approx_FSR(1, current_state_agent_1, join_action, number_drones, number_actions, env_shape)
                            #print(phi_BF1.shape)
                            #print(Theta.shape)
                            Q_i_1 = np.round(np.matmul(Theta[:, 0], phi_BF1)[0], 3)  # take real value. Matmul results in (1,) array. Also, rounding to 3 decimals.

                            phi_BF2 = Approx_FSR(2, current_state_agent_2, join_action, number_drones, number_actions, env_shape)
                            Q_i_2 = np.round(np.matmul(Theta[:, 1], phi_BF2)[0], 3)

                            phi_BF3 = Approx_FSR(3, current_state_agent_3, join_action, number_drones, number_actions, env_shape)
                            Q_i_3 = np.round(np.matmul(Theta[:, 2], phi_BF3)[0], 3)

                            CE = Q_i_1* Q_i_2 * Q_i_3


                            Q_A_i = np.array([possible_action_1, possible_action_2, possible_action_3, join_action, CE])

                            if Q_A.size == 0:
                                Q_A = Q_A_i
                            else:
                                Q_A = np.vstack((Q_A, Q_A_i))
    #print("Optimal: ", Q_A.size)
    # Determine the maximum one
    if Q_A.size != 0:
        #print("Optimal joint action solution found!")
        if Q_A[0].size > 1:
            Q_A_CE_array = Q_A[:, 4]

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
                action_set = Q_A[index_optimal_array[random_optimal_index], 0:3]    # Meaning take value from index 0, 1, 2 (so up to 3)
                # print "optimal action found! (multi)"
                # print Q_A_CE_array[random_optimal_index], max(Q_A_CE_array), min(Q_A_CE_array)

                return action_set
            elif index_optimal_array.size == 1: # one unique optimal value
                action_set = Q_A[index_optimal_array, 0:3]  # meaning take value from index 0, 1, 2 (so up to 3)
                # print "optimal action found! (one)"
                # print "unique CE", Q_A_CE_array
                return action_set
            else:   # none optimal value
                random_optimal_index = randint(0, Q_A_CE_array.size - 1)
                action_set = Q_A[index_optimal_array[random_optimal_index], 0:3]  # meaning take value from index 0, 1, 2 (so up to 3)
                return action_set
        else:
            action_set = Q_A[0:3]
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


def RandomJoinAction(V, number_actions, F, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, number_drones, num_actions, env_shape):
    current_state_agent_1 = V[0]
    current_state_agent_2 = V[1]
    current_state_agent_3 = V[2]

    Q_A = np.array([])

    '''
    action_set = Q_A
    action_set = []
    for drone in range(number_drones):
      action_set.append(np.random.choice(number_actions) + 1)
    action_set = np.array(action_set)
    return action_set
    '''
    #print(F)
    # Find possible value
    for possible_action_1 in range(1, number_actions+1):
        #print("before 1: ", current_state_agent_1, " ", possible_action_1)
        next_state_agent_1 = FindNextState(current_state_agent_1, possible_action_1, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, number_drones, num_actions, env_shape).astype(int)
        #print("1: " , next_state_agent_1)
        if (F[next_state_agent_1[0], next_state_agent_1[1]] > 0) and (not np.array_equal(next_state_agent_1, current_state_agent_1)):  # Evaluate selectable actions to eliminate inappropriate actions
            for possible_action_2 in range(1, number_actions+1):
                next_state_agent_2 = FindNextState(current_state_agent_2, possible_action_2, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, number_drones, num_actions, env_shape).astype(int)
                #print("2: " , next_state_agent_2)
                if (F[next_state_agent_2[0], next_state_agent_2[1]] > 0) and (not np.array_equal(next_state_agent_1, next_state_agent_2)) and (not np.array_equal(next_state_agent_2, current_state_agent_2)):
                    #print("2: " , next_state_agent_2)
                    for possible_action_3 in range(1, number_actions+1):
                        next_state_agent_3 = FindNextState(current_state_agent_3, possible_action_3, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, number_drones, num_actions, env_shape).astype(int)
                        #print("3: " , next_state_agent_3)
                        if (F[next_state_agent_3[0], next_state_agent_3[1]] > 0) and (not np.array_equal(next_state_agent_1, next_state_agent_3)) and (not np.array_equal(next_state_agent_2, next_state_agent_3)) and (not np.array_equal(next_state_agent_3, current_state_agent_3)):

                            Q_A_i = np.array([possible_action_1, possible_action_2, possible_action_3])
                            if Q_A.size == 0:
                                Q_A = Q_A_i
                            else:
                                Q_A = np.vstack((Q_A, Q_A_i))
    #print("Random: ", Q_A.size)
    # return a random action
    if Q_A.size != 0:
        if (Q_A[0].size > 1):   #check if an array is not 1D
            num_rows, num_cols = Q_A.shape
            random_index = np.random.randint(0, num_rows - 1)  # take one index randomly
            action_set = Q_A[random_index, 0:3]  # meaning take value from index 0, 1, 2 (so up to 3)
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

