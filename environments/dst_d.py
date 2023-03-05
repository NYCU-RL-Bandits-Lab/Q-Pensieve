from __future__ import absolute_import, division, print_function
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
from collections import deque
import numpy as np
from numpy import random
import heapq
import json
import os
import sys
import inspect
import math 
class treasure():
    def __init__(self, t_ID, pos,va):
        self.treasure_ID=t_ID
        self.position=pos
        self.value=va
    def reset(self):
        self.treasure_ID=0
        self.position=[0,0]
        self.value=va=0
class stone():
    def __init__(self, s_ID, pos):
        self.stone_ID=s_ID
        self.position=pos
    def reset(self):
        self.stone_ID=0
        self.position=[0,0]    
class DeepSeaTreasure(gym.Env):

    def __init__(self):
        # the map of the deep sea treasure (convex version)
        self.sea_map = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [-10, 8.2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [-10, -10, 11.5, 0, 0, 0, 0, 0, 0, 0, 0],
             [-10, -10, -10, 14.0, 15.1, 16.1, 0, 0, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, 19.6, 20.3, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, -10, -10, 0, 0, 0],
             [-10, -10, -10, -10, -10, -10, -10, -10, 22.4, 0, 0],
             [-10, -10, -10, -10, -10, -10, -10, -10, -10, 23.7, 0]]
        )

                
        # DON'T normalize
        self.max_reward = 1.0
        self.treasures = []
        self.stones = []
        self.steps=0
        t_c = 0
        s_c = 0
        for i in range(len(self.sea_map)):
            for j in range(len(self.sea_map[i])):  
                if(self.sea_map[i][j]>0.0):
                    self.treasures.append(treasure(t_c,[i+0.5,j+0.5],self.sea_map[i][j]))
                    t_c += 1
                if(self.sea_map[i][j]<0.0):
                    self.stones.append(stone(s_c,[i+0.5,j+0.5]))
                    s_c += 1              
        # state space specification: 2-dimensional discrete box
        self.state_spec = [['discrete', 1, [0, 11]], ['discrete', 1, [0, 11]]]
        self.observation_space = np.array([0,0])
        self.action_space = spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)
        # action space specification: 1 dimension, 0 up, 1 down, 2 left, 3 right
        self.action_spec = ['discrete', 1, [0, 4]]
    
        # reward specification: 2-dimensional reward
        # 1st: treasure value || 2nd: time penalty
        self.reward_spec = [[0, 14], [-1, 0]]
        
        self._max_episode_steps = 50
        self.max_episode_steps = 50
        self.reward_num = 2
        self.current_state = np.array([0, 0])
        self.terminal = False

    def get_map_value(self, pos):
        for i in range(len(self.treasures)):
            if (self.d_dis(pos,self.treasures[i].position,'t')<0.5):
                return self.treasures[i].value
        for i in range(len(self.stones)):
            if (self.d_dis(pos,self.stones[i].position,'s')<0.5):
                return -10
        return 0
    def d_dis(self,pos,o_pos,o_type):
        pos=np.array(pos)
        o_pos=np.array(o_pos)
        if(o_type == 't'):
            return np.sqrt(sum((o_pos-pos)*(o_pos-pos)))
        else:
            return np.max(np.abs(o_pos-pos))
    def reset(self):
        '''
            reset the location of the submarine
        '''
        self.current_state = np.array([0, 0])
        self.terminal = False
        self.steps = 0
        return self.current_state
    def reset_(self, state):
        '''
            reset the location of the submarine
        '''
        self.current_state = np.array(state)
        self.terminal = False
        self.steps = 0
        return self.current_state
    def step(self, action):
        '''
            step one move and feed back reward
        '''
        next_state = self.current_state + action

        valid = lambda x, ind: (x[ind] >= self.state_spec[ind][2][0]) and (x[ind] <= self.state_spec[ind][2][1])
        '''
        if valid(next_state, 0) and valid(next_state, 1):
            if self.get_map_value(next_state) != -1:
                self.current_state = next_state
        '''
        self.current_state = np.where( [ valid(next_state, 0 ), valid(next_state, 1) ], next_state, self.current_state)
            
        
        treasure_value = self.get_map_value(self.current_state)
        #print(treasure_value)
        if treasure_value == 0 or treasure_value == -1:
            treasure_value = 0.0
        else:
            treasure_value /= self.max_reward
            self.terminal = True
        time_penalty = -1.0 / self.max_reward
        reward = np.array([treasure_value, time_penalty])
        #reward = treasure_value + time_penalty
        self.steps += 1
        if(self.steps >= self._max_episode_steps):
            self.terminal = True
        return self.current_state, reward, self.terminal, {}
    
register(id='dst_d-v0', entry_point='environments.dst_d:DeepSeaTreasure')
