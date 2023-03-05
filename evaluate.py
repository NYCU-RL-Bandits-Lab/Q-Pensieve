import os
import numpy as np
import visdom
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from rltorch.memory import MultiStepMemory, PrioritizedMemory

from model import TwinnedQNetwork, GaussianPolicy
import random
import gym
from dst_d import DeepSeaTreasure



date = 'MOSAC-set4-buf0-seed2'
critic = TwinnedQNetwork(2,2,2,[256,256])
critic.load('./logs/dst_d-v0/'+date+'/model/critic_13.0.pth')

policy = GaussianPolicy(4,2,[256,256])
policy.load('./logs/dst_d-v0/'+date+'/model/policy_15.0.pth')

device = 'cuda'

vis = visdom.Visdom()
env = gym.make('dst_d-v0')


def q_heatmap(action,prefer):
    prefer = torch.tensor( prefer,dtype=torch.float32  )
    action = torch.tensor( action,dtype=torch.float32  )

    value = np.empty([11,11])
    time = np.empty([11,11])

    for i in range(11):
        for j in range(11):
            state = torch.tensor( np.array([[i,j]]),dtype=torch.float32 )

            c = critic(state,action,prefer)[0][0].detach().numpy()
            value[10-i][j] = c[0]
            time[10-i][j] = c[1]
            print(c)

    #vis.heatmap(X=value,)
    #vis.heatmap(X=time,)

def quiver(prefer):
    prefer = torch.tensor( prefer,dtype=torch.float32  )
    x_dir = np.empty([11,11])
    y_dir = np.empty([11,11])
    for i in range(11):
        for j in range(11):
            state = torch.tensor( np.array([[i,j]]),dtype=torch.float32 )
            
            _, _, a = policy.sample(state, prefer)
            a = a.detach().numpy()
            print(a)
            x_dir[10-i][j] = a[0][1]
            y_dir[10-i][j] = -a[0][0]
    vis.quiver(X=x_dir,Y=y_dir)

def explore(state, preference):
        # act with randomness
        state = torch.FloatTensor(state).unsqueeze(0)
        preference = torch.FloatTensor(preference).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = policy.sample(state, preference)
        return action.cpu().numpy().reshape(-1)

def exploit(state, preference):
        # act without randomness

        state = torch.FloatTensor(state).unsqueeze(0)
        preference = torch.FloatTensor(preference).unsqueeze(0)
        with torch.no_grad():
            _, _, action = policy.sample(state, preference)
        return action.cpu().numpy().reshape(-1)

def evaluate(preference):
    preference = torch.tensor( preference,dtype=torch.float32  )
    episodes = 1
    returns = np.empty((episodes,2))
    preference = np.array(preference)
    for i in range(episodes):
        state = env.reset()
        episode_reward = np.zeros(2)
        done = False
        trace = []
        actions = []
        while not done:
            #action = explore(state, preference)
            action = exploit(state, preference)

            trace.append(list(state))
            actions.append(list(action))
            next_state, reward, done = env.step(action)
            episode_reward += reward
            state = next_state


        returns[i] = episode_reward
        print('state', trace )
        print('action', actions ) 
    mean_return = np.mean(returns, axis=0)
    print('-' * 60)
    print(f'preference ', preference,
              f'reward:', mean_return)
    print('-' * 60)


PREF=[ [0.9, 0.1],[0.8 ,0.2],[0.7 ,0.3],[0.6 ,0.4],[0.5 ,0.5],[0.4 ,0.6],[0.3 ,0.7],[0.2 ,0.8],[0.1 ,0.9] ]


pref = np.array([[0.9,0.1]])
q_heatmap( np.array( [[1,1]] ),pref )
#quiver(pref)

#evaluate(np.array([0.9,0.1]))
'''
for i in PREF:
    pref = np.array([i])
    quiver(pref)
q_heatmap( np.array( [[1,1]] ),pref )
q_heatmap( np.array( [[1,0]] ),pref )
q_heatmap( np.array( [[1,-1]] ),pref )
q_heatmap( np.array( [[0,-1]] ),pref )
q_heatmap( np.array( [[-1,-1]] ),pref )
q_heatmap( np.array( [[-1,0]] ),pref )
q_heatmap( np.array( [[-1,1]] ),pref )
q_heatmap( np.array( [[0,1]] ),pref )
'''


