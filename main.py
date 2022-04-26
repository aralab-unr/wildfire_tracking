# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 21:12:41 2021

@author: kripa
"""

import argparse
import numpy as np
from fire_model_test import FireEnvironment
from itertools import product, combinations
from cvxopt import solvers, matrix
from copy import deepcopy
import math
import matplotlib.pyplot as plt
import pickle
import shutil
import os
import sys
import time
import asyncio
from random import randint

solvers.options['show_progress'] = False


def generate_phi(env_shape, action_space, n_drones):
    X, Y, Z = env_shape
    state_dim = (X + Y + Z) * n_drones

    drone_actions = np.arange(action_space)
    actions = {x: i for i, x in enumerate(product(*((drone_actions,) * n_drones)))}
    def phi(S, A):
        states = []
        for i in range(n_drones):
            x, y, z = S[i]
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
        action_slot = actions[A] * len(states)
        state[action_slot: action_slot + len(states)] = states
        return state.astype(int).reshape(-1, 1)
    return phi, state_dim * action_space ** n_drones

def generate_pi(env_shape, action_space, n_drones):
    def pi(phi, theta, S, eps=0.9):
        sample = np.random.random()
        if sample < eps:
            # random joint action
            return {drone: np.random.choice(action_space) for drone in range(n_drones)}

        actions = list(product(*((np.arange(action_space),) * n_drones)))
        action_values = []
        
        #t0 = time.time()
        for action in actions:
            A = []
            b = []
            G = []
            h = []
            c = np.zeros(n_drones * action_space)

            for i in range(n_drones):
                c[i * action_space + action[i]] -= phi(S, action).T.dot(theta[i])

            G.append(-1 * np.identity(n_drones * action_space))
            h.extend([0] * (n_drones * action_space))

            for i in range(n_drones):
                arr = np.zeros(n_drones * action_space)
                arr[i * action_space: (i + 1) * action_space] = 1
                A.append(arr)
                b.append(1)

            for i in range(n_drones):
                arr = np.zeros(n_drones * action_space)
                for a in range(action_space):
                    action_ = tuple(x if j != i else a for j, x in enumerate(action))
                    arr[i * action_space + action[i]] -= phi(S, action).T.dot(theta[i]) - phi(S, action_).T.dot(theta[i])
                G.append(arr)
                h.append(0)

            A = matrix(np.stack(A).astype(float))
            b = matrix(np.stack(b).astype(float))
            c = matrix(np.array(c).flatten().astype(float).reshape(-1, 1))
            G = matrix(np.vstack(G).astype(float))
            h = matrix(np.array(h).astype(float).reshape(-1, 1))

            solved = solvers.lp(c, G, h, A=A, b=b)
            sol = np.array(solved['x'])
            action_values.append(np.sum([phi(S, action).T.dot(theta[i]) * sol[i * action_space + action[i]] for i in range(n_drones)]))
        #print(time.time() - t0)
        action_values = np.array(action_values)
        # random tiebreaking among max value actions
        A = actions[np.random.choice(np.flatnonzero(action_values == action_values.max()))]
        return {drone: A[drone] for drone in range(n_drones)}
    return pi


def FSR(env_shape, action_space, n_drones, S, A):
    X, Y, Z = env_shape
    state_dim = (X + Y + Z) * n_drones

    drone_actions = np.arange(action_space)
    actions = {x: i for i, x in enumerate(product(*((drone_actions,) * n_drones)))}
    states = []
    for i in range(n_drones):
        x, y, z = S[i]
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
    action_slot = actions[A] * len(states)
    state[action_slot: action_slot + len(states)] = states
    return state.astype(int).reshape(-1, 1)
    #return phi, state_dim * action_space ** n_drones
    return state.astype(int).reshape(-1, 1), state_dim * action_space ** n_drones

def FindNextState(current_state_agent_i, next_action, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX):
    agent_i_x = current_state_agent_i[0]
    agent_i_y = current_state_agent_i[1]
    agent_i_z = current_state_agent_i[2]

    if next_action == 0:
        if agent_i_x + 1 <= X_MAX:
            agent_i_x_new = agent_i_x - 1
            next_state = np.array([agent_i_x_new, agent_i_y, agent_i_z])
        else:
            agent_i_x_new = X_MAX
            next_state = np.array([agent_i_x_new, agent_i_y, agent_i_z])

    elif next_action == 1:

        if agent_i_x - 1 >= X_MIN:
            agent_i_x_new = agent_i_x + 1
            next_state = np.array([agent_i_x_new, agent_i_y, agent_i_z])
        else:
            agent_i_x_new = X_MIN
            next_state = np.array([agent_i_x_new, agent_i_y, agent_i_z])
    elif next_action == 2:
        if agent_i_y + 1 <= Y_MAX:
            agent_i_y_new = agent_i_y + 1
            next_state = np.array([agent_i_x, agent_i_y_new, agent_i_z])
        else:
            agent_i_y_new = Y_MAX
            next_state = np.array([agent_i_x, agent_i_y_new, agent_i_z])
    elif next_action == 3:
        if agent_i_y - 1 >= Y_MIN:
            agent_i_y_new = agent_i_y + 1
            next_state = np.array([agent_i_x, agent_i_y_new, agent_i_z])
        else:
            agent_i_y_new = Y_MIN;
            next_state = np.array([agent_i_x, agent_i_y_new, agent_i_z])
    elif next_action == 4:
        if agent_i_z + 1 <= Z_MAX:
            agent_i_z_new = agent_i_z + 1
            next_state = np.array([agent_i_x, agent_i_y, agent_i_z_new])
        else:
            agent_i_z_new = Z_MAX
            next_state = np.array([agent_i_x, agent_i_y, agent_i_z_new])
    elif next_action == 5:
        if agent_i_z - 1 >= Z_MIN:
            agent_i_z_new = agent_i_z - 1
            next_state = np.array([agent_i_x, agent_i_y, agent_i_z_new])
        else:
            agent_i_z_new = Z_MIN
            next_state = np.array([agent_i_x, agent_i_y, agent_i_z_new])
    else:
        next_state = np.array([agent_i_x, agent_i_y, agent_i_z])

    return next_state


def FindOptimalJoinAction(V, number_actions, Theta, F, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, env_shape, action_space, n_drones,):
    current_state_agent_1 = V[0]
    current_state_agent_2 = V[1]
    current_state_agent_3 = V[2]

    num_rows, num_cols = V.shape
    number_drones = num_rows

    Q_A = np.array([])

    # Find possible value
    for possible_action_1 in range(1, number_actions+1):
        next_state_agent_1 = FindNextState(current_state_agent_1, possible_action_1, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX).astype(int)
        if (F[next_state_agent_1[0], next_state_agent_1[1]] > 0) and (not np.array_equal(next_state_agent_1, current_state_agent_1)):   # Evaluate selectable actions to eliminate inappropriate actions
            for possible_action_2 in range(1, number_actions+1):
                next_state_agent_2 = FindNextState(current_state_agent_2, possible_action_2, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX).astype(int)
                if (F[next_state_agent_2[0], next_state_agent_2[1]] > 0) and (not np.array_equal(next_state_agent_1, next_state_agent_2)) and (not np.array_equal(next_state_agent_2, current_state_agent_2)):
                    for possible_action_3 in range(1, number_actions+1):
                        next_state_agent_3 = FindNextState(current_state_agent_3, possible_action_3, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX).astype(int)
                        if (F[next_state_agent_3[0], next_state_agent_3[1]] > 0) and (not np.array_equal(next_state_agent_1, next_state_agent_3)) and (not np.array_equal(next_state_agent_2, next_state_agent_3)) and (not np.array_equal(next_state_agent_3, current_state_agent_3)):
                            join_action = (possible_action_1 - 1) * (number_actions**(number_drones - 1)) + (possible_action_2 - 1) * (number_actions**(number_drones - 2)) + possible_action_3

                            phi_BF1 = FSR(current_state_agent_1, join_action, number_drones, number_actions)
                            Q_i_1 = np.round(np.matmul(Theta[:, 0], phi_BF1)[0], 3)  # take real value. Matmul results in (1,) array. Also, rounding to 3 decimals.

                            phi_BF2 = FSR(current_state_agent_2, join_action, number_drones, number_actions)
                            Q_i_2 = np.round(np.matmul(Theta[:, 1], phi_BF2)[0], 3)

                            phi_BF3 = FSR(current_state_agent_3, join_action, number_drones, number_actions)
                            Q_i_3 = np.round(np.matmul(Theta[:, 2], phi_BF3)[0], 3)

                            CE = Q_i_1* Q_i_2 * Q_i_3


                            Q_A_i = np.array([possible_action_1, possible_action_2, possible_action_3, join_action, CE])

                            if Q_A.size == 0:
                                Q_A = Q_A_i
                            else:
                                Q_A = np.vstack((Q_A, Q_A_i))
    # Determine the maximum one

    if Q_A.size != 0:
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
        return action_set




def RandomJoinAction(V, number_actions, F, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX):
    current_state_agent_1 = V[0]
    current_state_agent_2 = V[1]
    current_state_agent_3 = V[2]

    Q_A = np.array([])

    # Find possible value
    for possible_action_1 in range(1, number_actions+1):
        next_state_agent_1 = FindNextState(current_state_agent_1, possible_action_1, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX).astype(int)
        if (F[next_state_agent_1[0], next_state_agent_1[1]] > 0) and (not np.array_equal(next_state_agent_1, current_state_agent_1)):  # Evaluate selectable actions to eliminate inappropriate actions
            for possible_action_2 in range(1, number_actions+1):
                next_state_agent_2 = FindNextState(current_state_agent_2, possible_action_2, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX).astype(int)
                if (F[next_state_agent_2[0], next_state_agent_2[1]] > 0) and (not np.array_equal(next_state_agent_1, next_state_agent_2)) and (not np.array_equal(next_state_agent_2, current_state_agent_2)):
                    for possible_action_3 in range(1, number_actions+1):
                        next_state_agent_3 = FindNextState(current_state_agent_3, possible_action_3, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX).astype(int)
                        if (F[next_state_agent_3[0], next_state_agent_3[1]] > 0) and (not np.array_equal(next_state_agent_1, next_state_agent_3)) and (not np.array_equal(next_state_agent_2, next_state_agent_3)) and (not np.array_equal(next_state_agent_3, current_state_agent_3)):

                            Q_A_i = np.array([possible_action_1, possible_action_2, possible_action_3])
                            if Q_A.size == 0:
                                Q_A = Q_A_i
                            else:
                                Q_A = np.vstack((Q_A, Q_A_i))

    # return a random action
    if Q_A.size != 0:
        if (Q_A[0].size > 1):   #check if an array is not 1D
            num_rows, num_cols = Q_A.shape
            random_index = randint(0, num_rows - 1)  # take one index randomly
            action_set = Q_A[random_index, 0:3]  # meaning take value from index 0, 1, 2 (so up to 3)
        else:
            action_set = Q_A
    else:
        action_set = Q_A
        print ("V=", current_state_agent_1, current_state_agent_2, current_state_agent_3)     # flags if empty array
    return action_set



def main():
    
    outputs_dir = "results"
    seed = 0
    np.random.seed(seed)
    n_drones = 3
    n_episodes = 700
    episode_max_steps = 2000
    min_eps = 0.05
    max_eps = 0.95
    eps_decay = 10000
    learning_rate = 0.1
    gamma = 0.9 
    dim_x = 11
    dim_y = 11
    dim_z = 5
    agents_theta_degrees = 30

    if os.path.exists(outputs_dir):
        shutil.rmtree(outputs_dir)

    os.makedirs(outputs_dir)


    env = FireEnvironment("fbndry4.txt", dim_x, dim_y, [dim_x, dim_y, dim_z], agents_theta_degrees, n_drones)
    action_space = env.action_space.n
    env_dim = tuple([x for x in env.map.shape] + [max(env.map.shape)])

    phi, phi_dim = generate_phi(env_dim, action_space, n_drones)
    theta = np.zeros((n_drones, phi_dim))

    pi = generate_pi(env_dim, action_space, n_drones)

    episode_rewards = np.zeros(n_episodes)
    episode_steps = np.zeros(n_episodes).astype(int)
    steps = 0
    file = "test2.txt"
    temp_t = []
    prev_sum = 0
    #temp_t.append(theta[0])
    reward_storage = []
    reward_counter = []
    reward_count = 0
    try:
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            #env.print(file)
            env_counter = 1
            for k in range(episode_max_steps):
                if((k + 1) % 5 == 0):
                    #print(k + 1)
                    env_counter +=1
                    env.grid = env.simStep(env_counter)
                    env.map = env.grid
                temp_reward = 0
                for i in range(n_drones):
                    epsilon = min_eps + (max_eps - min_eps) * math.exp(-1. * steps / eps_decay)
                    pi_A = pi(phi, theta, state, eps=epsilon)
                    next_state, reward, done, meta = env.step(pi_A)
                    #print(next_state)
                    A = tuple([pi_A[drone] for drone in range(n_drones)])
                    actions = product(*((np.arange(action_space),) * n_drones))
                    q_next = np.max([phi(next_state, action).T.dot(theta[i]) for action in actions])
                    #print(q_next)
                    theta[i] = theta[i] + learning_rate * (reward + gamma * q_next - phi(state, A).T.dot(theta[i])) * phi(state, A).flatten()
                    #if(i == 1):
                    #    temp_t.append(abs(np.sum(theta[0]) + np.sum(theta[1]) - prev_sum))
                    #    prev_sum = np.sum(theta[0]) + np.sum(theta[1])
                    episode_rewards[episode] += reward
                    temp_reward = reward
                    state = next_state
                    #env.print(file)
                    if done:
                        break
                reward_count += 1
                reward_counter.append(reward_count)
                reward_storage.append(temp_reward)
                episode_steps[episode] += 1
                steps += 1
                #plt.plot(reward_storage)
                #plt.pause(0.01)
                #print(steps)
                if done:
                    break
            print(f'Episode {episode}: {k + 1} steps.')
            #print(theta)
            #print(theta.size)
            #if(episode >= 2):
                #print(temp_t)
            #    plt.plot(temp_t)
            #    plt.show()
            #    sys.exit(0)            
            #env.print()
    except KeyboardInterrupt:
        pass
    #plt.ylim(0, episode_max_steps)
    plt.plot(reward_storage)
    plt.savefig(os.path.join(outputs_dir, 'episode_steps.png'))
    pickle.dump(episode_steps, open(os.path.join(outputs_dir, 'episode_steps.pkl'), 'wb'))

if __name__ == '__main__':
    main()