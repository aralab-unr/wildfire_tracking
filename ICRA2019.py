import numpy as np
import random
from random import randint
import matplotlib.pyplot as plt

# from keras.models import Sequential
# from keras.layers import *
# from keras.optimizers import *
#
#
# # function:
# def Approx_RBF2(current_state_agent_i, join_action, number_drones, number_actions):
#     model = Sequential()
#
#     model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1, 28, 28)))
#     model.add(Convolution2D(32, 3, 3, activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(10, activation='softmax'))
#
#     # 8. Compile model
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
#
#     # 9. Fit model on training data
#     model.fit(X_train, Y_train,
#               batch_size=32, nb_epoch=10, verbose=1)
#
#     return phi_BF


def Approx_RBF(current_state_agent_i, join_action, number_drones, number_actions):

    number_join_action = number_actions ** number_drones
    phi_BF = np.zeros((8 * number_join_action, 1), dtype=np.float64)  # Basis function: 8 RBF


    #print("state : \n", current_state_agent_i)
    #print(join_action)
    # Define center array of the RBF
    mu = 2
    center_array = np.zeros((8, 3))
    center_array[0] = [3, 3, 1]
    center_array[1] = [3, 3, 3]
    center_array[2] = [3, 5, 1]
    center_array[3] = [3, 5, 3]
    center_array[4] = [5, 3, 1]
    center_array[5] = [5, 3, 3]
    center_array[6] = [5, 5, 1]
    center_array[7] = [5, 5, 3]

    for i in range(0, 8):
        center_i = center_array[i]
        distance = np.linalg.norm(current_state_agent_i - center_i)
        phi_BF[(join_action - 1) * 8 + i] = np.exp((-distance**2) / (2 * mu**2))

    return phi_BF


def FindNextState(current_state_agent_i, next_action, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX):
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


def FindOptimalJoinAction(V, number_actions, Theta, F, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX):
    current_state_agent_1 = V[0]
    current_state_agent_2 = V[1]
    current_state_agent_3 = V[2]

    #print("1 : ", current_state_agent_1)
    #print("2 : ", current_state_agent_2)
    #print("3 : ", current_state_agent_3)

    num_rows, num_cols = V.shape
    number_drones = num_rows

    Q_A = np.array([])

    # Find possible value
    for possible_action_1 in range(1, number_actions+1):
        next_state_agent_1 = FindNextState(current_state_agent_1, possible_action_1, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX).astype(int)
        #print("next state: ", next_state_agent_1)
        if (F[next_state_agent_1[0], next_state_agent_1[1]] > 0) and (not np.array_equal(next_state_agent_1, current_state_agent_1)):   # Evaluate selectable actions to eliminate inappropriate actions
            for possible_action_2 in range(1, number_actions+1):
                next_state_agent_2 = FindNextState(current_state_agent_2, possible_action_2, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX).astype(int)
                if (F[next_state_agent_2[0], next_state_agent_2[1]] > 0) and (not np.array_equal(next_state_agent_1, next_state_agent_2)) and (not np.array_equal(next_state_agent_2, current_state_agent_2)):
                    for possible_action_3 in range(1, number_actions+1):
                        next_state_agent_3 = FindNextState(current_state_agent_3, possible_action_3, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX).astype(int)
                        if (F[next_state_agent_3[0], next_state_agent_3[1]] > 0) and (not np.array_equal(next_state_agent_1, next_state_agent_3)) and (not np.array_equal(next_state_agent_2, next_state_agent_3)) and (not np.array_equal(next_state_agent_3, current_state_agent_3)):
                            join_action = (possible_action_1 - 1) * (number_actions**(number_drones - 1)) + (possible_action_2 - 1) * (number_actions**(number_drones - 2)) + possible_action_3

                            phi_BF1 = Approx_RBF(current_state_agent_1, join_action, number_drones, number_actions)
                            #print(phi_BF1.shape)
                            #print(Theta.shape)
                            Q_i_1 = np.round(np.matmul(Theta[:, 0], phi_BF1)[0], 3)  # take real value. Matmul results in (1,) array. Also, rounding to 3 decimals.

                            phi_BF2 = Approx_RBF(current_state_agent_2, join_action, number_drones, number_actions)
                            Q_i_2 = np.round(np.matmul(Theta[:, 1], phi_BF2)[0], 3)

                            phi_BF3 = Approx_RBF(current_state_agent_3, join_action, number_drones, number_actions)
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

    #print("1 : ", current_state_agent_1)
    #print("2 : ", current_state_agent_2)
    #print("3 : ", current_state_agent_3)

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


def CheckIfPointBelongToFOV(q, state_agent_i, FOV_angle):

    c = np.array([state_agent_i[0], state_agent_i[1]])
    z = state_agent_i[2]

    # print q
    # print c
    # print (q - c)
    # print z*np.tan(np.deg2rad(FOV_angle))
    qc_dif = np.abs((q - c)) - z*np.tan(np.deg2rad(FOV_angle))
    # print qc_dif
    if all([x <= 10**(-7) for x in qc_dif]):      # Error = +/-10^-7
        Inside_FOV_i_flag = True
    else:
        Inside_FOV_i_flag = False

    return Inside_FOV_i_flag


def FindNeighbors(agent_i, V):

    r = 10**5 #radius

    num_rows, num_cols = V.shape
    number_drones = num_rows

    Ni = np.array([])
    qi = V[agent_i]

    for agent_j in range(0, number_drones):
        if agent_j != agent_i:
            qj = V[agent_j]
            distance = np.linalg.norm((qj - qi))
            if distance <= r:
                if Ni.size == 0:
                    Ni = np.array([agent_j])
                else:
                    Ni = np.hstack((Ni, np.array([agent_j])))
    return Ni


def EstimateGlobalReward(F, V, FOV_angle, X_MIN, X_MAX, Y_MIN, Y_MAX):

    num_rows, num_cols = V.shape
    number_drones = num_rows

    reward_array = np.zeros(number_drones)
    overlap_point_array = np.zeros(number_drones)
    total_point = 0

    for agent_i in range(0, number_drones):

        Ni = FindNeighbors(agent_i, V)

        overlap_point = 0
        number_point_under_FOV = 0
        number_overlap_points = 0

        current_state_agent_i = V[agent_i]

        for pos_under_FOV_y in range(Y_MIN, Y_MAX+1):
            for pos_under_FOV_x in range(X_MIN, X_MAX+1):

                if F[pos_under_FOV_x, pos_under_FOV_y] > 0:

                    if agent_i == 0:
                        total_point = total_point + 1

                    A_Point = np.array([pos_under_FOV_x, pos_under_FOV_y])

                    if CheckIfPointBelongToFOV(A_Point, current_state_agent_i, FOV_angle):

                        number_overlap_agents = 1
                        if Ni.size != 0:
                            for i in range(0, Ni.size):
                                agent_j = Ni[i]
                                state_agent_j = V[agent_j]
                                overlapped = 0
                                if CheckIfPointBelongToFOV(A_Point, state_agent_j, FOV_angle):
                                    number_overlap_points = number_overlap_points + 1
                                    overlapped = 1

                                if overlapped == 1:
                                    number_overlap_agents = number_overlap_agents + 1
                                    if number_overlap_agents == 2:
                                        overlap_point = overlap_point + 1

                        number_point_under_FOV = number_point_under_FOV + (1.0/number_overlap_agents)

        reward_array[agent_i] = number_point_under_FOV
        overlap_point_array[agent_i] = overlap_point

    # print V.flatten()
    # print "overlap_point_array", overlap_point_array
    # print "total_point", total_point
    # print "sum(reward_array) =", np.sum(reward_array)
    # Estimate the global rewards

    if (np.sum(reward_array) + 0.1 >= total_point) and (np.sum(overlap_point_array) <= 1): # overlapping == 1 still meaning ==0, because need at least 2
        CRi = np.array([1, 1, 1], dtype=np.float64)
        terminal_state = True
        print ("Terminal : ", CRi, np.sum(reward_array), total_point, np.sum(overlap_point_array))
    else:
        CRi = np.array([0, 0, 0], dtype=np.float64)
        terminal_state = False

    point_uncovered = total_point - np.sum(reward_array)
    point_covered = total_point - point_uncovered
    total_point_overlapped = np.sum(overlap_point_array) / 2

    # print "point uncovered:", point_uncovered
    # print "point_covered:", point_covered
    # print "total_point:", total_point
    return CRi, terminal_state, point_uncovered, point_covered, total_point_overlapped


# Main program
# Declare constant
fov_angle = np.array([29, 29])
number_drones = 3
drones_array = np.array([1, 2, 3], dtype=np.integer)

number_episodes = 700 # 700
number_time_steps = 2000 # 2000
learning_episode_run = np.linspace(1, number_episodes*number_time_steps, num=number_episodes*number_time_steps)

# Q-learning parameters
epsilon = 1
learning_rate = 0.1
discount_rate = 0.9

# Generate the environment
X_MAX = 7
X_MIN = 1
Y_MAX = 7
Y_MIN = 1
Z_MAX = 4
Z_MIN = 1     # z min = 1

F = np.zeros((8, 8))
high_intensity = 5
mid_intensity = 3
low_intensity = 1

F[3, 2] = low_intensity
F[4, 2] = low_intensity
F[2, 3] = low_intensity
F[5, 3] = low_intensity
F[3, 4] = low_intensity
F[6, 4] = low_intensity
F[3, 5] = low_intensity
F[5, 5] = low_intensity
F[4, 6] = low_intensity

F[3, 3] = low_intensity
F[4, 3] = low_intensity
F[4, 4] = low_intensity
F[5, 4] = low_intensity
F[4, 5] = low_intensity

F = np.genfromtxt("foi-0.csv", delimiter=',')

print (F)


# Define actions:
number_actions = 6
FORWARD_X = 1
BACK_X = 2
FORWARD_Y = 3
BACK_Y = 4
UP_Z = 5
DOWN_Z = 6
STAY = 7    # unused

# Main parameter
print(number_actions)
print(number_drones)
Theta = np.zeros((8*number_actions**number_drones, number_drones), dtype=np.float64)

# Start collaborative learning
episode = 0
failed_episode = 0

total_step_run = np.zeros(number_episodes)  # Step taken
try:
    for episode in range(0, number_episodes):
        print ("Episode ", episode)  # episode starts from 0
        Theta_backup = Theta    #backup Theta to restore if thing goes wrong
    
        # reduce the epsilon after each episode
        epsilon = epsilon*0.9
    
        # Initialize the drone position
        c1 = np.array((randint(1, 7), randint(1, 7), randint(1, 3))) # x, y, z
        while F[c1[0], c1[1]] <= 0:
            c1 = np.array((randint(1, 7), randint(1, 7), randint(1, 3)))
    
        c2 = np.array((randint(1, 7), randint(1, 7), randint(1, 3)))
        while F[c2[0], c2[1]] <= 0:
            c2 = np.array((randint(1, 7), randint(1, 7), randint(1, 3)))
    
        c3 = np.array((randint(1, 7), randint(1, 7), randint(1, 3)))
        while F[c3[0], c3[1]] <= 0:
            c3 = np.array((randint(1, 7), randint(1, 7), randint(1, 3)))
    
        V = np.array([c1, c2, c3])
        # print "initial state:", V
        # num_rows, num_cols = V.shape
        # print num_rows
    
        # Containers
        terminal_state = 0
        terminal_state_ever = np.zeros(number_episodes)
        local_goal_reached = np.zeros(number_drones)
    
        t = 0   # t start from 0 too
        while (t < number_time_steps) and (terminal_state == False):
    
            # Start: FCMARL-Q learning Algorithm
            p = random.uniform(0, 1)
            if p >= epsilon:
               A = FindOptimalJoinAction(V, number_actions, Theta, F, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX)
            else:
               A = RandomJoinAction(V, number_actions, F, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX)
    
            if A.size == 0:         # Prevent empty case
                terminal_state = True
                failed_episode = failed_episode + 1
                Theta = Theta_backup
                print ("Failed episode! Attempt to redo the episode")
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
                A =  {drone: A[drone] for drone in range(number_drones)}
                #print("ACtions mod: ", A)
                for agent_i in range(0, number_drones):
                    current_state_agent_i = V[agent_i]
                    a_prime = A[agent_i]
                    new_qi = FindNextState(current_state_agent_i, a_prime, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX)
                    Vnew[agent_i, 0:3] = new_qi
                #print(Vnew)
                # Estimate global reward
                [CRi, terminal_state, point_uncovered, point_covered, total_point_overlapped] = EstimateGlobalReward(F, Vnew, fov_angle, X_MIN, X_MAX, Y_MIN, Y_MAX)
                #print(CRi)
                # Find the next optimal
                A_next = FindOptimalJoinAction(Vnew, number_actions, Theta, F, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX)
    
                action_taken = int((A[0] - 1) * (number_actions ** (number_drones - 1)) + (A[1] - 1) * (number_actions ** (number_drones - 2)) + A[2])
    
                opt_join_action = int((A_next[0] - 1) * (number_actions ** (number_drones - 1)) + (A_next[1] - 1) * (number_actions ** (number_drones - 2)) + A_next[2])
    
                for agent_i in range(0, number_drones):
    
                    previous_state_agent_i = V[agent_i]     # Previous
    
                    new_state_agent_i = Vnew[agent_i]   # new
    
                    phi_BF_previous = Approx_RBF(previous_state_agent_i, action_taken, number_drones, number_actions)
              
                    phi_BF_new = Approx_RBF(new_state_agent_i, opt_join_action, number_drones, number_actions)
    
                    # Update function:
    
                    TD_diff_i = learning_rate * (CRi[agent_i] + np.round(discount_rate * np.matmul(Theta[:, agent_i], phi_BF_new)[0] - np.matmul(Theta[:, agent_i], phi_BF_previous)[0], 5))*phi_BF_previous
    
                    # if np.count_nonzero(TD_diff_i) != 0:
                        # print "CRi = ", CRi[agent_i], "*BF_new = ", discount_rate * np.matmul(Theta[:, agent_i], phi_BF_new)[0], "BF_previous = ", np.matmul(Theta[:, agent_i], phi_BF_previous)[0], "difference = ", discount_rate * np.matmul(Theta[:, agent_i], phi_BF_new)[0] - np.matmul(Theta[:, agent_i], phi_BF_previous)[0]
    
                    Theta[:, agent_i:agent_i+1] = Theta[:, agent_i:agent_i+1] + TD_diff_i   # slicing the array to directly manipulate the data.
    
                # Update new position and step
                # print "next state:", Vnew
                # print "A = ", A
                # print "A_next =", A_next
    
                V = Vnew
                t = t + 1
    
                if terminal_state == 1:
                    terminal_state_ever[episode] = 1
    
        print ("Non-zero values of Theta = ", np.count_nonzero(Theta))
        # print Theta[:, 0]
        # num_rows, num_cols = Theta[:, 0].shape
        # print num_rows
        # print num_cols
    
        # print np.matmul(np.transpose(phi_BF_new), Theta[:, 0])
        # print TD_diff_i
        # print Theta[:, 1]
    
    # Plot the total step
except KeyboardInterrupt:
    fig, ax = plt.subplots()
    ax.plot(total_step_run, 'k--', label='Total Step Taken')
    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('#00FFCC')
    plt.show()
