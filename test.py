from model import TwinnedQNetwork, GaussianPolicy
import dst_d
import gym
import torch
import numpy as np
import half_cheetah_v3
import hopper_v3
import ant_v3
import walker2d_v3

import math
import os


import argparse

os.environ["CUDA_VISIBLE_DEVICES"]="3"
cuda = True
device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

def compute_hv(objs, ref_point):
    x, hv = ref_point[0], 0.0
    for i in range(len(objs)):
        hv += (max(ref_point[0], objs[i][0]) - x) * (max(ref_point[1], objs[i][1]) - ref_point[1])
        x = max(ref_point[0], objs[i][0])
    return hv


def explore(state, preference):
        # act with noisy
        
        state = torch.FloatTensor(state).unsqueeze(0)
        preference = torch.FloatTensor(preference).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = policy.sample(state, preference)
        return action.cpu().numpy().reshape(-1)

def exploit(state, preference):
        # act without policy.load(f'./logs/{env_name}/{date}/model/policy_{args.model_id}.0.pth')

        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        preference = preference.clone().detach().to(device).unsqueeze(0)
        with torch.no_grad():
            _, _, action = policy.sample(state, preference)
        return action.cpu().numpy().reshape(-1)


parser = argparse.ArgumentParser()

parser.add_argument('--env_id', type=str, default='MO_hopper-v0')
parser.add_argument('--set_num', type=int, default=8)
parser.add_argument('--buf_num', type=int, default=0)
parser.add_argument('--q_freq', type=int, default=1000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--model_id', type=int, default=15)
parser.add_argument('--ref_point', type=float, nargs='+', default=[0., 0.])
args = parser.parse_args()

env_name = args.env_id



ref = args.ref_point
    
env = gym.make(env_name)
env.seed(args.seed)
reward_num = env.reward_num

policy = GaussianPolicy(env.observation_space.shape[0] + env.reward_num ,env.action_space.shape[0],[256,256]).to(device)
date = 'sac-seed0-20210522-1950'
date = 'sac-seed710-20210720-2013'

a = np.arange(0,1,0.01)
table = np.stack((1-a,a),-1)

hv = []
utility = []


frac = 1

date = f"MOSAC-set{args.set_num}-buf{args.buf_num}-seed{args.seed}_freq{args.q_freq}"
policy.load(f'./logs/{env_name}/{date}/model/policy_{args.model_id}.0.pth')

state=env.reset()
env.continuous = True
step = 0

#preference = torch.tensor( [0.45,0.45,0.1],dtype=torch.float32  )

epi = 0
epi_num = 10
total_reward_vec = np.zeros( (table.shape[0], env.reward_num))
total_reward = np.zeros( table.shape[0])
for i in range(table.shape[0]):#perf 0.01~1
    state,done = env.reset(),False
    p = table[i]
    preference = torch.tensor( p,dtype=torch.float32  )
    episode_rewards = np.zeros((epi_num ,reward_num))
    episode_reward = np.zeros(reward_num)

    for j in range(epi_num):
        done = False
        while not done:
            action = exploit(state, preference)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            step += 1
            if done:
                state=env.reset()
                print('reward', p)
                episode_rewards[j] = episode_reward
                print(np.dot(episode_reward,p),episode_reward)
                print('='*70)
                episode_reward = np.zeros(reward_num)

                step = 0
    print('==='*70)
    m = np.mean(episode_rewards,0)
    print(m,p)
    total_reward_vec[i] = m
    total_reward[i] = np.dot(m,p)
    print('==='*70)
total_reward_vec = total_reward_vec[total_reward_vec[:, 0].argsort()]
h = compute_hv(total_reward_vec, ref)#hv

hv.append(h)
utility.append(np.mean(total_reward))


print(date)
print('Hyper volume')
print(hv)
print(np.mean(hv))
print('utility')
print(utility)
print(np.mean(utility))
