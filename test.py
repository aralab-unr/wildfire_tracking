import os
import numpy as np
import random
from random import randint
import matplotlib.pyplot as plt
import copy

global_counter = 0


def actionFunc(i, all_actions, print_vector):
  global global_counter
  for j in range(len(all_actions[i])):
    t_vector = copy.copy(print_vector)
    if(True and i < (len(all_actions) - 1)):
      #global_counter = global_counter + 1
      t_vector.append(all_actions[i][j])
      actionFunc(i + 1, all_actions, t_vector)
    elif True:
      global_counter = global_counter + 1
      t_vector.append(all_actions[i][j])
      print(t_vector)
    



n_actions = 6
n_drones = 4 


all_actions = []

for i in range(n_drones):
  t_actions = []
  for j in range(1, n_actions + 1):
    t_actions.append(j)
  all_actions.append(t_actions)

print(all_actions)

i = 0
for j in range(len(all_actions[i])):
  print_vector = []
  if(True):
    #global_counter = global_counter + 1
    print_vector.append(all_actions[i][j])
    actionFunc(i + 1, all_actions, print_vector)

print("Global Counter: " , global_counter)