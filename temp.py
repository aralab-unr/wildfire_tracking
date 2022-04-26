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
from utility import Approx_FSR, FindNextState, FindOptimalJoinAction, RandomJoinAction
from fire_model_test import FireEnvironment

solvers.options['show_progress'] = False


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


    np.random.seed(seed)

    env = FireEnvironment("fbndry4.txt", dim_x, dim_y, [dim_x, dim_y, dim_z], agents_theta_degrees, n_drones)
    action_space = env.action_space.n
    env_dim = tuple([x for x in env.map.shape] + [max(env.map.shape)])


    #phi, phi_dim = generate_phi(env_dim, action_space, args.n_drones)
    n_drones = 3    
    num_drones = n_drones
    number_drones = n_drones
    X, Y, Z = env_dim
    state_dim = (X + Y + Z)
    action_space = 6
    number_actions = action_space
    theta = np.zeros((n_drones, state_dim * action_space ** n_drones))

    Theta = np.zeros((state_dim * action_space ** n_drones, number_drones), dtype=np.float64)
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
    X_MAX = 10
    X_MIN = 0
    Y_MAX = 10
    Y_MIN = 0
    Z_MAX = 4
    Z_MIN = 1     # z min = 1
    total_step_run = np.zeros(number_episodes)     
    fov_angle = np.array([30, 30])
    failed_episode = 0    
    np.random.seed(0)
    try:
        for episode in range(n_episodes):
            Theta_backup = theta
            epsilon = epsilon * 0.9
            state = env.reset()
            #env.grid = env.simStep(env_counter)
            env.map = env.grid
            done = False
            V = np.asarray(state)
            terminal_state = 0
            terminal_state_ever = np.zeros(number_episodes)
            local_goal_reached = np.zeros(number_drones)
            for k in range(episode_max_steps):
                #epsilon = args.min_eps + (args.max_eps - args.min_eps) * math.exp(-1. * steps / args.eps_decay)
                #pi_A = pi(phi, theta, state, eps=epsilon)
                A = None
                p = np.random.uniform(0, 1)
                #print(p)
                if(p >= epsilon):
                  A = FindOptimalJoinAction(V, number_actions, Theta, env.map, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, n_drones, action_space, env_dim)
                  print("Optimal")
                else:
                  A = RandomJoinAction(V, number_actions, env.map, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, n_drones, action_space, env_dim)
                  print("random")
                
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
                    # initialize containers
                    Vnew = np.zeros((number_drones, 3))
                    agent_reward_t = np.array([])
                    total_reward_t = 0
                    total_action_t = 0

                    # count step until convergence
                    total_step_run[episode] = total_step_run[episode] + 1

                    #print("Actions: " , A)
                    temp_A =  {drone: A[drone] - 1 for drone in range(number_drones)}
                    #print("Actions mod: ", A)

                    next_state, reward, done, meta = env.step(temp_A)
                    Vnew = np.asarray(next_state) 
                    CRi = np.array([0, 0, 0], dtype=np.float64)
                    if(done):
                      CRi = np.array([1, 1, 1], dtype=np.float64)
                            
                    print("~~~~~~~~~~~~~~~~")
                    print("state: ", state)
                    print("Actions: ", A)                    
                    print("Next State: ", next_state)
                    print("~~~~~~~~~~~~~~~~")

                    A_next = FindOptimalJoinAction(Vnew, number_actions, Theta, env.map, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, n_drones, action_space, env_dim)
                    print(A_next)
                    action_taken = int((A[0] - 1) * (number_actions ** (number_drones - 1)) + (A[1] - 1) * (number_actions ** (number_drones - 2)) + A[2])
                    opt_join_action = int((A_next[0] - 1) * (number_actions ** (number_drones - 1)) + (A_next[1] - 1) * (number_actions ** (number_drones - 2)) + A_next[2])
                    
                    for agent_i in range(0, number_drones):
                        previous_state_agent_i = V[agent_i] 
                        new_state_agent_i = Vnew[agent_i]
                        phi_BF_previous = Approx_FSR(previous_state_agent_i, action_taken, number_drones, number_actions, env_dim)
                        phi_BF_new = Approx_FSR(new_state_agent_i, opt_join_action, number_drones, number_actions, env_dim)
                        TD_diff_i = learning_rate * (CRi[agent_i] + np.round(discount_rate * np.matmul(Theta[:, agent_i], phi_BF_new)[0] - np.matmul(Theta[:, agent_i], phi_BF_previous)[0], 5))*phi_BF_previous
                        Theta[:, agent_i:agent_i+1] = Theta[:, agent_i:agent_i+1] + TD_diff_i 
                  
                    #q_next = np.max([phi(next_state, action).T.dot(theta[i]) for action in actions])
                    #theta[i] = theta[i] + args.lr * (reward + args.gamma * q_next - phi(state, A).T.dot(theta[i])) * phi(state, A).flatten()
                  
                    episode_rewards[episode] += reward
                    state = next_state
                    V = Vnew

                    if done:
                        break
                episode_steps[episode] += 1
                steps += 1
            print(f'Episode {episode}: {k + 1} steps.')
    except KeyboardInterrupt:
        pass
    plt.ylim(0, episode_max_steps)
    plt.plot(episode_steps)
    plt.savefig(os.path.join(outputs_dir, 'episode_steps.png'))
    pickle.dump(episode_steps, open(os.path.join(outputs_dir, 'episode_steps.pkl'), 'wb'))

if __name__ == '__main__':
    main()