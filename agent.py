import os
import visdom
import numpy as np
import torch
import copy
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from rltorch.memory import MultiStepMemory, PrioritizedMemory
from base import QMemory

from model import TwinnedQNetwork, GaussianPolicy
from utils import grad_false, hard_update, soft_update, to_batch,\
    update_params, RunningMeanStats
import random
from multi_step import *
from datetime import datetime
import time

#PREF = [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8],[0.1,0.9]]

p_name= ['9505','9010','8515','8020','7525','7030','6535','6040','5545','5050','4555','4060','3565','3070','2575','2080','1585','1090','0595']
PREF = [[0.95,0.05],[0.9, 0.1], [0.85, 0.15], [0.8, 0.2], [0.75, 0.25], [0.7, 0.3], [0.65, 0.35], [0.6, 0.4], [0.55, 0.35], [0.5, 0.5], [0.45, 0.55], [0.4, 0.6], [0.35, 0.65], [0.3, 0.7], [0.25, 0.75],[0.2, 0.8], [0.15,0.85] ,[0.1,0.9]]
class QMonitor(object):
    def __init__(self,train=True):
        #self.f = open('Q4.txt', 'w')
        a=1
    def update(self, eps, a, b, c, d):
        a=1
        #print(f'{a} {b} {c} {d}',file=self.f)
        
class Monitor(object):

    def __init__(self, spec,path):

        env_name = spec['env_name']
        set_num = spec['set_num']
        buf_num = spec['buf_num']
        self.vis = visdom.Visdom(env = f'MOSAC{datetime.now().strftime("%m%d")}-{env_name}_pref{set_num}_buf{buf_num}' ,port = 8097)
        self.spec = spec
        if spec['pref'][0] == 0.9:
            print(999)
            self.path = os.path.join(path,'reward_log91.npz')
        elif spec['pref'][0] == 0.5:
            print(555)
            self.path = os.path.join(path,'reward_log55.npz')
        elif spec['pref'][0] == 0.1:
            print(111)
            self.path = os.path.join(path,'reward_log19.npz')


        self.value_window = None
        self.text_window = None

    def update(self, eps, tot_reward, Rew_1, Rew_2, loss):

        if self.value_window == None:
            self.tot_t = np.array([tot_reward])
            self.rew_1_t = np.array([Rew_1])
            self.rew_2_t = np.array([Rew_2])
            self.loss_t = np.array([loss])
            self.value_window = self.vis.line(X=torch.Tensor([eps]).cpu(),
                                              Y=torch.Tensor([tot_reward, Rew_1, Rew_2, loss]).unsqueeze(0).cpu(),
                                              opts=dict(xlabel='steps_per10000',
                                                        ylabel='Reward value',
                                                        title='Value Dynamics ' + str(self.spec['pref']) + ' ' + str(self.spec['seed']),
                                                        legend=['Total Reward', 'forward_reward', 'ctrl cost','loss']))
        else:
            #Smoothing
            self.tot_t = np.append(self.tot_t, tot_reward)
            tot_reward = np.mean(self.tot_t[-20:])
            
            self.rew_1_t = np.append(self.rew_1_t, Rew_1)
            Rew_1 = np.mean(self.rew_1_t[-20:])
            
            self.rew_2_t = np.append(self.rew_2_t, Rew_2)
            Rew_2 = np.mean(self.rew_2_t[-20:])
            
            if hasattr(self, 'path'):
                np.savez(self.path,tot=self.tot_t, rew_1 = self.rew_1_t, rew_2 = self.rew_2_t)
            
            self.loss_t = np.append(self.loss_t, loss)
            loss = np.mean(self.loss_t[-20:])

            self.vis.line(
                X=torch.Tensor([eps]).cpu(),
                Y=torch.Tensor([tot_reward, Rew_1, Rew_2, loss]).unsqueeze(0).cpu(),
                win=self.value_window,
                update='append')

class SacAgent:

    def __init__(self, env, log_dir, num_steps=3000000, batch_size=256, 
                 lr=0.0003, hidden_units=[256, 256], memory_size=1e6, prefer_num = 8, buf_num = 0,
                 gamma=0.99, tau=0.005, entropy_tuning=True, ent_coef=0.2,
                 multi_step=1, per=False, alpha=0.6, beta=0.4,
                 beta_annealing=0.0001, grad_clip=None, updates_per_step=1,
                 start_steps=10000, log_interval=10, target_update_interval=1,
                 eval_interval=1000, cuda=True, seed=0, cuda_device=0, q_frequency=1000):
        self.env = env

        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True  # It harms a performance.
        torch.backends.cudnn.benchmark = False
        self.q_frequency = q_frequency
        self.QM = QMonitor()
        
        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")
        print(self.device)
        print(self.env.observation_space.shape[0])
        print(self.env.reward_num)
        self.policy = GaussianPolicy(
            self.env.observation_space.shape[0]+self.env.reward_num,
            self.env.action_space.shape[0],
            hidden_units=hidden_units).to(self.device)
        self.critic = TwinnedQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            self.env.reward_num,
            hidden_units=hidden_units).to(self.device)
        self.critic_target = TwinnedQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            self.env.reward_num,
            hidden_units=hidden_units).to(self.device).eval()

        # copy parameters of the learning network to the target network
        hard_update(self.critic_target, self.critic)
        # disable gradient calculations of the target network
        grad_false(self.critic_target)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.critic.Q2.parameters(), lr=lr)

        if entropy_tuning:
            # Target entropy is -|A|.
            self.target_entropy = -torch.prod(torch.Tensor(
                self.env.action_space.shape).to(self.device)).item()
            # We optimize log(alpha), instead of alpha.
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=lr)
        else:
            # fixed alpha
            self.alpha = torch.tensor(ent_coef).to(self.device)

        if per:
            # replay memory with prioritied experience replay
            # See https://github.com/ku2482/rltorch/blob/master/rltorch/memory
            self.memory = PrioritizedMemory(
                memory_size, self.env.observation_space.shape,
                self.env.action_space.shape, self.device, gamma, multi_step,
                alpha=alpha, beta=beta, beta_annealing=beta_annealing)
        else:

            # replay memory without prioritied experience replay
            # See https://github.com/ku2482/rltorch/blob/master/rltorch/memory
            self.memory = MOMultiStepMemory(
                memory_size, self.env.observation_space.shape, self.env.reward_num,
                self.env.action_space.shape, self.device, gamma, multi_step)

        #Q Replay Buffer
        self.Q_memory = QMemory(buf_num)
        self.cur_p = 0
        self.cur_e = 0
        self.qmem_p = 0
        self.qmem_e = 0

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        
        self.monitor = []
        self.tot_t = []
        self.reward_v = []
        if self.env.reward_num == 3:
            PREF_=np.load("3pref_table.npy")
        elif self.env.reward_num == 4:
            PREF_=np.load("4pref_table.npy")
        elif self.env.reward_num == 5:
            PREF_=np.load("5pref_table.npy")
        else:
            PREF_ = PREF
        for i in PREF_:
            self.tot_t.append([])
            self.reward_v.append([])

        
        self.set_num = prefer_num # set of ω'
        
        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.num_steps = num_steps
        self.tau = tau
        self.per = per
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.gamma_n = gamma ** multi_step
        self.entropy_tuning = entropy_tuning
        self.grad_clip = grad_clip
        self.updates_per_step = updates_per_step
        self.log_interval = log_interval
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval

    def get_pref(self):
        preference = np.random.rand( self.env.reward_num)
        preference = preference.astype(np.float32)
        preference /= preference.sum()
        return preference


    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return len(self.memory) > self.batch_size and\
            self.steps >= self.start_steps

    def act(self, state, preference=None):
        if preference is None:
            #rand = random.randint(0, len(PREF)-1)
            #preference = np.array(PREF[rand])
            preference = self.get_pref()
        if self.start_steps > self.steps:
            action = self.env.action_space.sample()
        else:
            action = self.explore(state,preference)
        return action

    def explore(self, state, preference):
        # act with randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        preference = torch.FloatTensor(preference).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.policy.sample(state, preference)
        return action.cpu().numpy().reshape(-1)

    def exploit(self, state, preference):
        # act without randomness
 
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        preference = torch.FloatTensor(preference).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, action = self.policy.sample(state, preference)
        return action.cpu().numpy().reshape(-1)

    def calc_current_q(self, states, preference, actions, rewards, next_states, dones):

        curr_q1, curr_q2 = self.critic(states, actions, preference)
        

        return curr_q1, curr_q2

    def calc_target_q(self, states, preference, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_entropies, _ = self.policy.sample(next_states, preference)
            next_q1, next_q2 = self.critic_target(next_states, next_actions, preference)           
            
            #We choose argmin_Q (ωTQ)
            w_q1 = torch.einsum('ij,j->i',[next_q1, preference[0] ])
            w_q2 = torch.einsum('ij,j->i',[next_q2, preference[0] ])
            mask = torch.lt(w_q1,w_q2)
            mask = mask.repeat([1,self.env.reward_num])
            mask = torch.reshape(mask, next_q1.shape)

            minq = torch.where( mask, next_q1, next_q2)
                
            next_q = minq + self.alpha * next_entropies

        target_q = rewards + (1.0 - dones) * self.gamma_n * next_q

        return target_q

    def train_episode(self):
        self.episodes += 1
        episode_reward = 0.
        episode_steps = 0
        done = False
        state = self.env.reset()

        #Sample preference from prefernence space
        preference = self.get_pref()
        if self.env.reward_num == 3:
            PREF_=np.load("3pref_table.npy")
        elif self.env.reward_num == 4:
            PREF_=np.load("4pref_table.npy")
        elif self.env.reward_num == 5:
            PREF_=np.load("5pref_table.npy")
        else:
            PREF_ = PREF
        while not done:
            ## Just fixed
            action = self.act(state, preference)
            #action = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            self.steps += 1
            episode_steps += 1
            episode_reward += reward

            # ignore done if the agent reach time horizons
            # (set done=True only when the agent fails)
            if episode_steps >= self.env.max_episode_steps:
                masked_done = False
            else:
                masked_done = done

            if self.per:
                batch = to_batch(
                    state, preference, action, reward, next_state, masked_done,
                    self.device)
     
                with torch.no_grad():
                    curr_q1, curr_q2 = self.calc_current_q(*batch)
                target_q = self.calc_target_q(*batch)
                error = torch.abs(curr_q1 - target_q).item()
                # We need to give true done signal with addition to masked done
                # signal to calculate multi-step rewards.
                self.memory.append(
                    state, preference, action, reward, next_state, masked_done, error,
                    episode_done=done)
            else:
                # We need to give true done signal with addition to masked done
                # signal to calculate multi-step rewards.

                self.memory.append(
                    state, preference, action, reward, next_state, masked_done,
                    episode_done=done)

            if self.is_update():
                for _ in range(self.updates_per_step):
                    self.learn()

            if self.steps % self.eval_interval == 0:
                for i in range(len(PREF_)):
                    #self.evaluate(PREF_[i],self.monitor[i],i)
                    self.evaluate_(PREF_[i],i)
                if self.steps % 100000 == 0:
                    self.save_models(self.steps/100000)

            state = next_state

        # We log running mean of training rewards.
        # self.train_rewards.append(episode_reward)

        
        print(f'episode: {self.episodes:<4}  '
              f'episode steps: {episode_steps:<4}  '
              f'episode weight: {preference}  '
              f'reward:', episode_reward)

    def learn(self):
        self.learning_steps += 1
        if self.learning_steps % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        if self.learning_steps % self.q_frequency == 0 and self.learning_steps > 20000:
            co = copy.deepcopy(self.critic)
            self.Q_memory.append(co)
        
        if self.per:
            # batch with indices and priority weights
            batch, indices, weights = \
                self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
            # set priority weights to 1 when we don't use PER.
            weights = 1.


        rand = random.randint(0, len(PREF)-1)
        PREF_SET = []
        # Form preference set W containing the updating preference
        preference = self.get_pref()
        preference = torch.tensor(preference ,device = self.device)
        PREF_SET.append(preference)
        for _ in range(self.set_num-1):
            p = self.get_pref()
            p = torch.tensor(p ,device = self.device)
            PREF_SET.append(p)

               
        
        q1_loss, q2_loss, errors, mean_q1, mean_q2 =\
            self.calc_critic_loss(batch, weights, preference, PREF_SET)
        
        policy_loss, entropies = self.calc_policy_loss(batch, weights, preference, PREF_SET)

        update_params(
            self.q1_optim, self.critic.Q1, q1_loss, self.grad_clip)
        update_params(
            self.q2_optim, self.critic.Q2, q2_loss, self.grad_clip)
        update_params(
            self.policy_optim, self.policy, policy_loss, self.grad_clip)

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropies, weights)
            update_params(self.alpha_optim, None, entropy_loss)
            self.alpha = self.log_alpha.exp()
        if self.per:
            # update priority weights
            self.memory.update_priority(indices, errors.cpu().numpy())
        
        self.QM.update(self.steps, self.cur_p, self.cur_e, self.qmem_p, self.qmem_e)
    def calc_critic_loss(self, batch, weights, preference, PREF):
        

        states, _, actions, rewards, next_states, dones = batch

        q1_losses = []
        q2_losses = []
        errorses = []
        mean_q1s = []
        mean_q2s = []

        
        D_pref = preference.repeat(self.batch_size,1)

        curr_q1, curr_q2 = self.calc_current_q(states, D_pref, actions, rewards, next_states, dones)
        
    
        target_q = self.calc_target_q(states, D_pref, actions, rewards, next_states, dones)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)
        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()
      

        # Critic loss is mean squared TD errors with priority weights.
        q1_loss = torch.mean(torch.tensordot((curr_q1 - target_q).pow(2), preference,dims=1) * weights)
        q2_loss = torch.mean(torch.tensordot((curr_q2 - target_q).pow(2), preference,dims=1) * weights)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights, preference, PREF):
        start = time.time()
        states, _, actions, rewards, next_states, dones = batch
        preference_batch = preference.repeat(self.batch_size, 1)
        
        losses = []

        c_cnt = 0
        for a, c in enumerate([ self.critic]+self.Q_memory.sample() ): # Use critic from Q Replay Buffer
            for b, i in enumerate(PREF): #Get Q from preference set W
                p_batch = torch.tensor(i, device = self.device).repeat(self.batch_size, 1)
                sampled_action, entropy, _ = self.policy.sample(states, p_batch)
                if a == 0 and b == 0:
                    e = entropy
                q1, q2 = c(states, sampled_action, preference_batch)
                
                q1 = torch.tensordot(q1, preference, dims = 1)
                q2 = torch.tensordot(q2, preference, dims = 1)
                q = torch.min(q1, q2)
                
                l = - q - self.alpha * entropy
                losses.append(l)

        losses = torch.stack(losses, dim = 1)
        policy_loss, idx =  torch.min(losses, 1)
        ll=idx.detach().cpu()[:,0].tolist()
        policy_loss = torch.mean(policy_loss)

        
        sampled_action, e, _ = self.policy.sample(states, preference_batch)

        return policy_loss, e

    def calc_entropy_loss(self, entropy, weights):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropy).detach()
            * weights)
        return entropy_loss

    def evaluate_(self, preference, ind):
        episodes = 10
        returns = np.empty((episodes,self.env.reward_num))
        preference = np.array(preference)
        for i in range(episodes):
            state = self.env.reset()
            episode_reward = np.zeros(self.env.reward_num)
            done = False
            trace = []
            actions = []
            while not done:
                action = self.exploit(state,preference )
                trace.append(list(state))
                actions.append(list(action))
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state

            returns[i] = episode_reward
            print(episode_reward)
        mean_return = np.mean(returns, axis=0)
        
        batch = self.memory.sample(self.batch_size) 
        p = torch.tensor(preference ,device = self.device, dtype=torch.float32)
        with torch.no_grad():
            q1_loss, q2_loss, errors, mean_q1, mean_q2 =\
                            self.calc_critic_loss(batch, 1, p, 0)
        #monitor.update(self.steps/self.eval_interval, np.dot(preference,mean_return), *mean_return, q1_loss.mean().item())


        path = os.path.join(self.log_dir, 'summary')
        tot_path = os.path.join(path, f'{ind}total_log.npy')
        reward_path = os.path.join(path, f'{ind}reward_log.npy')
        self.tot_t[ind].append( np.dot(preference, mean_return) )
        self.reward_v[ind].append(mean_return)

        np.save(tot_path, np.array(self.tot_t[ind]) )
        np.save(reward_path, np.array(self.reward_v[ind]) )

        print('-' * 60)
        print(f'preference ', preference,
              f'Num steps: {self.steps:<5}  '
              f'reward:', mean_return)
        print('-' * 60)

    def evaluate(self, preference, monitor, ind):
        episodes = 10
        returns = np.empty((episodes,self.env.reward_num))
        preference = np.array(preference)
        for i in range(episodes):
            state = self.env.reset()
            episode_reward = np.zeros(self.env.reward_num)
            done = False
            trace = []
            actions = []
            while not done:
                action = self.exploit(state,preference )
                trace.append(list(state))
                actions.append(list(action))
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state

            returns[i] = episode_reward
            print(episode_reward)
        mean_return = np.mean(returns, axis=0)
        
        batch = self.memory.sample(self.batch_size) 
        p = torch.tensor(preference ,device = self.device, dtype=torch.float32)
        with torch.no_grad():
            q1_loss, q2_loss, errors, mean_q1, mean_q2 =\
                            self.calc_critic_loss(batch, 1, p, 0)
        #monitor.update(self.steps/self.eval_interval, np.dot(preference,mean_return), *mean_return, q1_loss.mean().item())


        path = os.path.join(self.log_dir, 'summary')
        tot_path = os.path.join(path, f'{p_name[ind]}total_log.npy')
        reward_path = os.path.join(path, f'{p_name[ind]}reward_log.npy')
        self.tot_t[ind].append( np.dot(preference, mean_return) )
        self.reward_v[ind].append(mean_return)

        np.save(tot_path, np.array(self.tot_t[ind]) )
        np.save(reward_path, np.array(self.reward_v[ind]) )

        print('-' * 60)
        print(f'preference ', preference,
              f'Num steps: {self.steps:<5}  '
              f'reward:', mean_return)
        print('-' * 60)

    def save_models(self, num):
        self.policy.save(os.path.join(self.model_dir, 'policy_'+str(num)+'.pth'))
        self.critic.save(os.path.join(self.model_dir, 'critic_'+str(num)+'.pth'))
        self.critic_target.save(
            os.path.join(self.model_dir, 'critic_target.pth'))

    def __del__(self):
        #self.writer.close()
        self.env.close()
