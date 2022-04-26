# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 18:35:54 2021

@author: kripa
"""


#adapted and copied from https://github.com/eczy/rl-drone-coverage/blob/master/field_coverage_env.py

import gym
import numpy as np
from enum import Enum
import sys
import cv2
import matplotlib.pyplot as plt
import copy
import mahotas
from skimage.draw import polygon
import time



class FireEnvironment(gym.Env):
    
    class Action(Enum):
        Right = 1
        Left = 2
        North = 3
        South = 4
        Up = 5
        Down = 6  
    
    class FireInfo(object):
        def __init__(self, fire_id, x, y, intensity, agent_assigned):
            self.fire_id = fire_id
            self.x = x
            self.y = y
            self.intensity = intensity        
            self.agent_assigned = agent_assigned
            
    class Agent(object):
        def __init__(self, pos, fov):
            self.fov = fov
            self.pos = pos            
            
            
            
            
    def __init__ (self, data_file_name, height, width, shape, theta, num_agents,  X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, max_steps=2000):
        super().__init__()
        assert len(shape) == 3, 'Environmnet shape must be of form (X, Y, Z).'
        self.action_space = gym.spaces.Discrete(len(self.Action))
        self.observation_space = gym.spaces.Box(low=0, high=np.array(shape))        
        self.shape = shape
        self.theta = np.radians(theta)
        self.num_agents = num_agents
        self.max_steps = max_steps 
        self.height = height
        self.width = width
        self.file_name = data_file_name
        self.fire_data = self.readFireData(self.file_name)
        
        self.agents = {}
        self.steps = 0

        self.X_MIN = X_MIN
        self.X_MAX = X_MAX
        self.Y_MIN = Y_MIN
        self.Y_MAX = Y_MAX
        self.Z_MIN = Z_MIN
        self.Z_MAX = Z_MAX
        
        #print(len(self.fire_data))

        self.grid = None
        self.map = None
        #for x in range(200):
        #plt.ion()
        #self.fig, self.ax = plt.subplots()
        self.reset(0)
        
        #plt.xlim(0, shape[0] - 1)
        #plt.ylim(0, shape[0] - 1)
        t_agents = []
        self.points = None
        for x in range(len(self.agents)):
            t_agents.append((self.agents[x].pos[0], self.agents[x].pos[1], self.agents[x].pos[2]))
        self.t_agents = np.asarray(t_agents)   
        #self.sc = self.ax.scatter(self.t_agents[:, 0], self.t_agents[0:, 1])     
        #self.step({1:0, 2:1, 3:2})
        #print(self.map)
        

    def readFireData(self, file_name):
        f = open(file_name, "r")
        return_data = []
        while True:
            data = f.readline()
            if not data:
                break
            vals = data.split(",")
            if(len(vals) > 1):
                return_data.append(vals)
                
        return return_data

    def simStep(self, time_t):
        img = np.zeros((self.height,self.width), np.uint8)
        fire_id = 0
        fire_map = {}
        vals = self.fire_data[time_t]
        fire_info = []
        for j in range(0,len(vals)-1,3):
            #x = int(vals[j]) - 23 #20
            #print(j, " ", vals[j])
            #y = int(vals[j+1]) - 23 #22
            x = int(vals[j]) - 19
            y = int(vals[j+1]) - 19
            intensity = vals[j+2]
            intensity = 1.0 
            #j += 3 
            if str(str(x) + "," + str(y)) in fire_map:
                pass
            else:
               fire_map[str(x) + "," + str(y)] = intensity
               temp = self.FireInfo(int(fire_id), int(x), int(y), int(intensity), False)
               fire_id += 1
               fire_info.append(temp)
        poly_points = []
        
        for t in fire_info:
            #print(t.x, " ", t.y)
            img = cv2.circle(img, (t.x, t.y), 1, (255,255,255), -1)
            poly_points.append((t.x, t.y))
            
        
        poly_fill = cv2.fillPoly(img, np.int32([poly_points]), (255, 255, 255))
        points = np.transpose(np.where(poly_fill==255))
        self.points = np.asarray(points)
        #print(self.points)
        #self.points[:, 0] = self.shape[0] - self.points[:, 0] 
        #self.points[:, 1] = self.shape[1] - self.points[:, 1] 
        fire_map.clear()
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #print(img)
        return img    


    def reset(self, simulation_counter):
        self.steps = 0
        X, Y, Z = self.shape 
        agents = {}
        positions = set()
        self.grid = self.simStep(simulation_counter)
        #self.grid = np.genfromtxt("foi-0.csv", delimiter=',')
        #self.grid = np.float32(self.grid)
        self.map = self.grid        
        for i in range(self.num_agents):
            pos = np.random.randint(self.X_MIN, self.X_MAX), np.random.randint(self.Y_MIN, self.Y_MAX), np.random.randint(self.Z_MIN, self.Z_MAX - 1)
            while True:
                pos = np.random.randint(self.X_MIN, self.X_MAX), np.random.randint(self.Y_MIN, self.Y_MAX), np.random.randint(self.Z_MIN, self.Z_MAX - 1)
                if (pos not in agents.values()) and (self.map[pos[0], pos[1]] > 0):
                    break
            positions.add(pos)
            agents[i] = self.Agent(pos, self.theta)
        self.agents = agents
        #self.showAnimation()
        return self.state()
    
    def showAnimation(self):
        t_agents = []
        for x in range(len(self.agents)):
            t_agents.append((self.agents[x].pos[0], self.agents[x].pos[1], self.agents[x].pos[2]))

        self.t_agents = np.asarray(t_agents)
        t_c = []
        for t_c_c in range(len(self.agents)):
          t_c.append('r')
        colors = np.asarray(t_c)
        colors_field = np.ones(len(self.points)) 
        plt.clf()
        plt.xlim(0, self.shape[0] - 1)
        plt.ylim(0, self.shape[0] - 1)
        
        t_verts = []
        for x in range(len(self.agents)):
            verts = []
            t_agents.append((self.agents[x].pos[0], self.agents[x].pos[1], self.agents[x].pos[2]))
            x_proj = np.tan(self.agents[x].fov) * self.agents[x].pos[2]
            y_proj = np.tan(self.agents[x].fov) * self.agents[x].pos[2]
            x1 = (self.agents[x].pos[0]) - x_proj
            x2 = (self.agents[x].pos[0]) + x_proj
            y1 = (self.shape[0] - 1 - self.agents[x].pos[1]) - y_proj
            y2 = (self.shape[0] - 1 - self.agents[x].pos[1]) + y_proj
            verts.append([x1, y1])
            verts.append([x2, y1])
            verts.append([x2, y2])
            verts.append([x1, y2])
            verts.append([x1, y1])
            t_verts.append(verts)


        verts = np.array(t_verts)
        for x in range(len(self.agents)):
          plt.plot(verts[x, :, 0], verts[x, :, 1],'k-')
        #plt.plot(verts[1, :, 0], verts[1, :, 1],'k-')
        #plt.plot(verts[2, :, 0], verts[2, :, 1],'k-')
        #print(self.points)
        plt.scatter(self.points[:, 1], self.shape[0] - self.points[:, 0] - 1, marker='.', s=150, c=colors_field, alpha=1.0)
        plt.scatter(self.t_agents[:,0], self.shape[0] - 1 - self.t_agents[:, 1], marker='+', s=150, c=colors, alpha=1.0)
        self.fig.canvas.draw_idle()
        
    def state(self):
        return [x.pos for x in self.agents.values()]
    
    def step(self, action):
        #print(len(action))
        #plt.clf()
        assert len(action) == len(self.agents), 'Joint action must be defined for each agent.'
        temp_img = self.map.copy()
        color = (255, 0, 0)
        thickness = -1
        temp = cv2.cvtColor(temp_img, cv2.COLOR_GRAY2BGR)
        for drone, a in action.items():
            self.move_drone(drone, self.Action(a))
            temp = cv2.circle(temp, (self.agents[drone].pos[1], self.agents[drone].pos[0]), 1, color, thickness)

        observation = self.state()
        reward = self.reward()
        #success = reward > 0
        success = reward > 0.99
        done = success or self.steps == self.max_steps
        #temp = cv2.cvtColor(temp_img, cv2.COLOR_GRAY2BGR)
        cv2.namedWindow("simulation", cv2.WINDOW_NORMAL)
        #print(temp.shape)
        temp = cv2.resize(temp, (1000,1000), interpolation = cv2.INTER_AREA)
        #changed on february 2022
        cv2.imshow("simulation", temp)
        cv2.waitKey(1)
        '''
        t_agents = []
        for x in range(len(self.agents)):
            t_agents.append((self.agents[x].pos[0], self.agents[x].pos[1], self.agents[x].pos[2]))

        self.t_agents = np.asarray(t_agents)
        colors = np.array(['r', 'r', 'r'])
        colors_field = np.ones(len(self.points)) 
        plt.clf()
        plt.xlim(0, 7)
        plt.ylim(0, 7)
        
        t_verts = []
        for x in range(len(self.agents)):
            verts = []
            t_agents.append((self.agents[x].pos[0], self.agents[x].pos[1], self.agents[x].pos[2]))
            x_proj = np.tan(self.agents[x].fov) * self.agents[x].pos[2]
            y_proj = np.tan(self.agents[x].fov) * self.agents[x].pos[2]
            x1 = self.agents[x].pos[0] - x_proj
            x2 = self.agents[x].pos[0] + x_proj
            y1 = self.agents[x].pos[1] - y_proj
            y2 = self.agents[x].pos[1] + y_proj
            verts.append([x1, y1])
            verts.append([x2, y1])
            verts.append([x2, y2])
            verts.append([x1, y2])
            verts.append([x1, y1])
            t_verts.append(verts)


        verts = np.array(t_verts)
        plt.plot(verts[0, :, 0], verts[0, :, 1],'k-')
        plt.plot(verts[1, :, 0], verts[1, :, 1],'k-')
        plt.plot(verts[2, :, 0], verts[2, :, 1],'k-')
        plt.scatter(self.points[:, 0], self.points[:, 1], marker='.', s=150, c=colors_field, alpha=1.0)
        plt.scatter(self.t_agents[:, 0], self.t_agents[:,1], marker='+', s=150, c=colors, alpha=1.0)
        self.fig.canvas.draw_idle()
        '''
        #self.showAnimation()
        #plt.pause(0.1)
        return observation, reward, done, {'success!': success}
      
    
    def render(self):
        temp_img = self.map.copy()
        color = (255, 0, 0)
        thickness = -1
        temp = cv2.cvtColor(temp_img, cv2.COLOR_GRAY2BGR)
        for drone in range(len(self.agents)):
            temp = cv2.circle(temp, (self.agents[drone].pos[0], self.agents[drone].pos[1]), 1, color, thickness)    
        
        temp = cv2.resize(temp, (220,220), interpolation = cv2.INTER_AREA)
        cv2.imshow("filled", temp)
        cv2.waitKey(1)
            
    
    def reward(self):
        masks = self.view_masks()
        foi = self.map.astype(int)
        foi_orig = foi.copy()
        coverage = 0
        for i, drone in self.agents.items():
            coverage += np.sum(masks[i].flatten() & foi.flatten())
            foi = foi - (masks[i].flatten() & foi.flatten()).reshape(self.shape[0], self.shape[1])
        drones = set(self.agents.keys(),)
        overlap = 0
        if len(drones) > 1:
            for i, drone in self.agents.items():
                mask = masks[i]
                other_masks = np.sum([masks[x] for x in drones - {i}], axis=0)
                overlap += np.sum(mask.flatten() & other_masks.flatten())
                #print(drone.pos[2])
                #print(masks[i])
        #if coverage == sum(foi_orig.flatten()) and overlap == 0:
        #    return 1.0
        #return 0 
        #return(float(float(coverage)/float(sum(foi_orig.flatten()))))
        #if(overlap == 0):
        return(float(float(coverage)/float(sum(foi_orig.flatten()))))
        #return 0
    
    def view_masks(self):
        coordsx, coordsy = np.meshgrid(*[np.arange(x) for x in self.map.shape])
        view_masks = {}
        for i, drone in self.agents.items():
            mask = np.zeros_like(self.map).astype(int)
            x, y, z = drone.pos

            for xc, yc in zip(coordsx.flatten(), coordsy.flatten()):
                x_proj = np.tan(drone.fov) * z
                y_proj = np.tan(drone.fov) * z
                #print(z, " ", x_proj, " ", y_proj)
                if all([
                    xc > x - x_proj,
                    xc < x + x_proj,
                    yc > y - y_proj,
                    yc < y + y_proj
                ]):
                    mask[xc, yc] = True
            view_masks[i] = mask
        return view_masks    

    def move_drone(self, drone, action):
        X, Y, Z = self.shape
        x, y, z = self.agents[drone].pos

        if action == self.Action.Left:
            new_pos = max(x - 1, 0), y, z
        elif action == self.Action.Right:
            new_pos = min(x + 1, self.X_MAX), y, z
        elif action == self.Action.North:
            new_pos = x, min(y + 1, self.Y_MAX), z
        elif action == self.Action.South:
            new_pos = x, max(y - 1, 0), z
        elif action == self.Action.Up:
            new_pos = x, y, min(z + 1, self.Z_MAX)
        elif action == self.Action.Down:
            new_pos = x, y, max(z - 1, 1)
        else:
            raise ValueError(f'Invalid action {action} for agent {drone}')

        if new_pos in set(self.state()):
            return
        self.agents[drone].pos = new_pos    
        
    def print(self, file):
        f =  open(file, 'a+')
        coordsx, coordsy = np.meshgrid(*[np.arange(x) for x in self.map.shape])
        view_masks = {}
        for i, drone in self.agents.items():
            mask = np.zeros_like(self.map).astype(int)
            x, y, z = drone.pos
            x_proj = 0
            y_proj = 0
            for xc, yc in zip(coordsx.flatten(), coordsy.flatten()):
                x_proj = np.tan(drone.fov) * z
                y_proj = np.tan(drone.fov) * z
            
            #print(i, x, y, z, x_proj, y_proj)
            f.write(str(str(i) + " " + str(x) + " " + str(y) + " " + str(z) + " " + str(x_proj) + " " + str(y_proj)) + "\n")
        f.close()
    
 
    
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
    dim_x = 15
    dim_y = 15
    dim_z = 3
    agents_theta_degrees = 30 
    X_MAX = 14
    X_MIN = 0
    Y_MAX = 14
    Y_MIN = 0
    Z_MAX = 2
    Z_MIN = 0     # z min = 1

    env = FireEnvironment("fbndry4.txt", dim_x, dim_y, [dim_x, dim_y, dim_z], agents_theta_degrees, n_drones, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX)
    for i in range(200):
        env.grid = env.simStep(i)
        env.map = env.grid
        env.render()
        print(sum(env.map.flatten()))
        print(env.map)
     
if __name__ == "__main__":
    main()     