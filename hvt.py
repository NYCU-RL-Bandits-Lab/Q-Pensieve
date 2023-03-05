from model import TwinnedQNetwork, GaussianPolicy
from MO_lunar_lander import LunarLanderContinuous
import dst_d
import gym
import torch
import numpy as np
import half_cheetah_v3
import hopper_v3
import ant_v3

import math
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def compute_hv(objs, ref_point):
    x, hv = ref_point[0], 0.0
    for i in range(len(objs)):
        hv += (max(ref_point[0], objs[i][0]) - x) * (max(ref_point[1], objs[i][1]) - ref_point[1])
        x = max(ref_point[0], objs[i][0])
    return hv

def get_pref(reward_dim):
    preference = np.random.rand( reward_dim)
    preference = preference.astype(np.float32)
    preference /= preference.sum()
    return preference

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

env_name = "MO_hopper-v0"
env = gym.make(env_name)
env.seed(0)

policy = GaussianPolicy(env.observation_space.shape[0] + env.reward_num ,env.action_space.shape[0],[256,256])


gamma=0.99

hv = []
utility = []
for i in range(5):
    date = f"MOSAC-set4-buf4-seed{i}"
    policy.load(f'./logs/{env_name}/{date}/model/policy_15.0.pth')

    state=env.reset()
    env.continuous = True
    step = 0

    #preference = torch.tensor( [0.45,0.45,0.1],dtype=torch.float32  )
    p = [0,1]
    print(p)
    preference = torch.tensor( p,dtype=torch.float32  )


    episode_reward = np.zeros(env.reward_num)
    act = []
    sta = []
    tot = np.zeros((100, env.reward_num))
    epi = 0
    while epi<100:

        sta.append(list(state))
        action = exploit(state, preference)
        next_state, reward, done, _ = env.step(action)
        act.append(list(action))
        #env.render()
        #episode_reward += reward*math.pow(gamma, step)
        episode_reward += reward
        state = next_state
        step += 1
        if done:
            done=False
            state=env.reset()
            print('')
            print('reward', p)
            tot[epi] = episode_reward
            print(np.dot(episode_reward,p),episode_reward)
            print('='*70)
            utility.append(np.dot(episode_reward,p))
            p = [(epi+1)/100,1-(epi+1)/100]
            preference = torch.tensor( p,dtype=torch.float32  )
            act = []
            sta = []
            episode_reward = np.zeros(env.reward_num)
            step = 0
            epi += 1
    tot = tot[tot[:, 0].argsort()]

    print(tot)
    h = compute_hv(tot, np.array([0,-3000]) )
    print(h)
    hv.append(h)

#tot = np.array(tot)
#tot = np.mean(tot,0)

print(hv)
print(np.mean(hv))
print("utility", np.mean(utility))
