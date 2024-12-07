import numpy as np
import random
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import importlib
import wandb

from collections import deque
from stable_baselines3.common.buffers import ReplayBuffer
#from torch.utils.tensorboard import SummaryWriter
import Network
import Args
importlib.reload(Network)
importlib.reload(Args)
from Network import *
from functions import *
import gymnasium as gym
import Args

# Random Agent
class RandomAgent:
    def __init__(self, args:Args):
        self.args=args
        self.env = gym.make(args.env_id)
        self.episodes_lengths=[]
        self.rewards=[]
        self.sum_rewards=0
        self.accumulative_rewards=[]

    def select_action(self):        
        return random.randint(0, self.env.action_space.n - 1)  
    
    def train(self):
        obs, _ = self.env.reset()
        for self.global_step in range(self.args.total_timesteps):
            action = self.select_action()
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            obs=next_obs
            if done:
                self.episodes_lengths.append(self.global_step)
                obs, _ = self.env.reset()
            self.rewards.append(reward)
            self.sum_rewards+=reward
            self.accumulative_rewards.append(self.sum_rewards)
        return self.rewards,self.accumulative_rewards,self.episodes_lengths

# Q-learning Agent
class QLearningAgent:
    def __init__(self, args:Args):
        self.args=args
        self.gamma=0.95
        self.env = gym.make(args.env_id)
        
        # set-up
        if args.env_id=="FrozenLake-v1":
            self.q_table=np.full((self.env.observation_space.n, self.env.action_space.n),0.9)
        else:
            self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        
        self.D={}
        self.Z={}
        
        #resutls
        self.episodes_lengths=[]
        self.rewards=[]
        self.sum_rewards=0
        self.accumulative_rewards=[]

    def select_action(self,obs):
        self.D,epsilon=update_D(obs,self.D)
        if np.random.rand() <= epsilon:
            return random.randint(0, self.env.action_space.n - 1) 
        return np.argmax(self.q_table[obs]) 
    
    def learn(self, obs, action, reward, next_obs, done):
        self.Z,alpha=update_Z(obs,action,self.Z)
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_obs])
        
        self.q_table[obs, action] += alpha * (target - self.q_table[obs, action])   
    def train(self):
        obs, _ = self.env.reset()
        for self.global_step in range(self.args.total_timesteps):
            action = self.select_action(obs)
            
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.learn(obs,action,reward,next_obs,done)
            obs=next_obs
            if done:
                self.episodes_lengths.append(self.global_step)
                obs, _ = self.env.reset()
            self.rewards.append(reward)
            self.sum_rewards+=reward
            self.accumulative_rewards.append(self.sum_rewards)
        return self.rewards,self.accumulative_rewards,self.episodes_lengths

# Q-learning Stochastic
class Stoch_QLearningAgent:
    def __init__(self, args:Args):
        self.args=args
        self.gamma=0.95
        self.env = gym.make(args.env_id)
        self.C=self.args.C 
        
        # set-up
        if args.env_id=="FrozenLake-v1":
            self.q_table=np.full((self.env.observation_space.n, self.env.action_space.n),0.9)
        else:
            self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        
        self.D={}
        self.Z={}
        self.states_history={}
        
        # results
        self.episodes_lengths=[]
        self.rewards=[]
        self.sum_rewards=0
        self.accumulative_rewards=[]

    def select_action(self,obs):
        self.D,epsilon=update_D(obs,self.D)
        if np.random.rand() <= epsilon:
            action=random.randint(0, self.env.action_space.n - 1) 
            return action
        
        Action_subset=Subset_function(self.env.action_space.n,self.states_history[obs],self.C)
        action=Action_subset[np.argmax(self.q_table[obs,Action_subset])]
        return action
    
    def learn(self, obs, action, reward, next_obs, done):
        self.Z,alpha=update_Z(obs,action,self.Z)
        target = reward
        Action_subset=Subset_function(self.env.action_space.n,self.states_history[obs],self.C)
        if not done:
            target += self.gamma * np.max(self.q_table[next_obs,Action_subset])
        
        self.q_table[obs, action] += alpha * (target - self.q_table[obs, action])
            
            
    def train(self):
        obs, _ = self.env.reset()
        for self.global_step in range(self.args.total_timesteps):
            action = self.select_action(obs)
            
            if obs not in self.states_history:
                self.states_history[obs] = deque(maxlen=self.args.memory_size)
            self.states_history[obs].append(action)
            
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.learn(obs,action,reward,next_obs,done)
            obs=next_obs
            if done:
                self.episodes_lengths.append(self.global_step)
                obs, _ = self.env.reset()
            self.rewards.append(reward)
            self.sum_rewards+=reward
            self.accumulative_rewards.append(self.sum_rewards)
        return self.rewards,self.accumulative_rewards,self.episodes_lengths

# Double Q-Learning Agent#
class DoubleQLearningAgent:
    def __init__(self, args:Args):
        self.args=args
        self.gamma=0.95
        self.env = gym.make(args.env_id)
        if args.env_id=="FrozenLake-v1":
            self.q_table1=np.full((self.env.observation_space.n, self.env.action_space.n),0.9)
            self.q_table2=np.full((self.env.observation_space.n, self.env.action_space.n),0.9)
        else:
            self.q_table1 = np.zeros((self.env.observation_space.n, self.env.action_space.n))
            self.q_table2 = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        
        self.D={}
        self.ZA={}
        self.ZB={}
        # results
        self.episodes_lengths=[]
        self.rewards=[]
        self.sum_rewards=0
        self.accumulative_rewards=[]
        
    def select_action(self, obs):
        self.D,epsilon=update_D(obs,self.D)
        if np.random.rand() <= epsilon:
            return random.randint(0, self.env.action_space.n - 1)  
        q_sum = self.q_table1[obs] + self.q_table2[obs]
        return np.argmax(q_sum)

    def learn(self, obs, action, reward, next_obs, done):
        if np.random.rand() < 0.5:
            self.ZA,alpha=update_Z(obs,action,self.ZA)
            target = reward
            if not done:
                next_action = np.argmax(self.q_table1[next_obs])
                target += self.gamma * self.q_table2[next_obs, next_action]
            self.q_table1[obs, action] += alpha * (target - self.q_table1[obs, action])
        else:
            self.ZB,alpha=update_Z(obs,action,self.ZB)
            target = reward
            if not done:
                next_action = np.argmax(self.q_table2[next_obs])
                target += self.gamma * self.q_table1[next_obs, next_action]
            self.q_table2[obs, action] += alpha * (target - self.q_table2[obs, action])
 
    def train(self):
        obs, _ = self.env.reset()
        for self.global_step in range(self.args.total_timesteps):
            action = self.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.learn(obs,action,reward,next_obs,done)
            obs=next_obs
            if done:
                self.episodes_lengths.append(self.global_step)
                obs, _ = self.env.reset()
            self.rewards.append(reward)
            self.sum_rewards+=reward
            self.accumulative_rewards.append(self.sum_rewards)
        return self.rewards,self.accumulative_rewards,self.episodes_lengths
      
# Stochastic Double Q-Learning Agent#
class Stoch_DoubleQLearningAgent:
    def __init__(self, args:Args):
        self.args=args
        self.gamma=0.95
        self.env = gym.make(args.env_id)
        self.C=self.args.C
        
        # set-up
        if args.env_id=="FrozenLake-v1":
            self.q_table1=np.full((self.env.observation_space.n, self.env.action_space.n),0.9)
            self.q_table2=np.full((self.env.observation_space.n, self.env.action_space.n),0.9)
        else:
            self.q_table1 = np.zeros((self.env.observation_space.n, self.env.action_space.n))
            self.q_table2 = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.D={}
        self.ZA={}
        self.ZB={}
        self.states_history={}
        # results
        self.episodes_lengths=[]
        self.rewards=[]
        self.sum_rewards=0
        self.accumulative_rewards=[]
        
    def select_action(self, obs):
        self.D,epsilon=update_D(obs,self.D)
        if np.random.rand() <= epsilon:
            action=random.randint(0, self.env.action_space.n - 1)  
            return action
        
        q_sum = self.q_table1[obs] + self.q_table2[obs]
        
        Action_subset=Subset_function(self.env.action_space.n,self.states_history[obs],self.C)
        action=Action_subset[np.argmax(q_sum[Action_subset])]
        return action

    def learn(self, obs, action, reward, next_obs, done):
        Action_subset=Subset_function(self.env.action_space.n,self.states_history[obs],self.C)
        if np.random.rand() < 0.5:
            self.ZA,alpha=update_Z(obs,action,self.ZA)
            target = reward
            if not done:
                next_action = np.argmax(self.q_table1[next_obs,Action_subset])
                target += self.gamma * self.q_table2[next_obs, next_action]
            self.q_table1[obs, action] += alpha * (target - self.q_table1[obs, action])
        else:
            self.ZB,alpha=update_Z(obs,action,self.ZB)
            target = reward
            if not done:
                next_action = np.argmax(self.q_table2[next_obs,Action_subset])
                target += self.gamma * self.q_table1[next_obs, next_action]
            self.q_table2[obs, action] += alpha * (target - self.q_table2[obs, action])
 
    def train(self):
        obs, _ = self.env.reset()
        for self.global_step in range(self.args.total_timesteps):
            action = self.select_action(obs)
            
            if obs not in self.states_history:
                self.states_history[obs] = deque(maxlen=self.args.memory_size)
            self.states_history[obs].append(action)
            
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.learn(obs,action,reward,next_obs,done)
            obs=next_obs
            if done:
                self.episodes_lengths.append(self.global_step)
                obs, _ = self.env.reset()
            self.rewards.append(reward)
            self.sum_rewards+=reward
            self.accumulative_rewards.append(self.sum_rewards)
        return self.rewards,self.accumulative_rewards,self.episodes_lengths
                
#Sarsa Agent         
class SARSAAgent:
    def __init__(self, args:Args):
        self.args=args
        self.gamma=0.95
        self.env = gym.make(args.env_id)
        
        if args.env_id=="FrozenLake-v1":
            self.q_table=np.full((self.env.observation_space.n, self.env.action_space.n),0.9)
        else:
            self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.D={}
        self.Z={}
        # results
        self.episodes_lengths=[]
        self.rewards=[]
        self.sum_rewards=0
        self.accumulative_rewards=[]
        
    def select_action(self,obs):
        self.D,epsilon=update_D(obs,self.D)
        if np.random.rand() <= epsilon:
            return random.randint(0, self.env.action_space.n - 1) 
        
        return np.argmax(self.q_table[obs]) 
    
    def learn(self, obs, action, reward, next_obs,next_action, done):
        self.Z,alpha=update_Z(obs,action,self.Z)
        target = reward
        if not done:
            target += self.gamma * self.q_table[next_obs, next_action]
        
        self.q_table[obs, action] += alpha * (target - self.q_table[obs, action])   
        
    def train(self):
        obs, _ = self.env.reset()
        action=self.select_action(obs)
        
        for self.global_step in range(self.args.total_timesteps):
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            next_action=self.select_action(next_obs)
            self.learn(obs,action,reward,next_obs,next_action,done)
            obs=next_obs
            action=next_action
            if done:
                self.episodes_lengths.append(self.global_step)
                obs, _ = self.env.reset()
                
            self.rewards.append(reward)
            self.sum_rewards+=reward
            self.accumulative_rewards.append(self.sum_rewards)
        return self.rewards,self.accumulative_rewards,self.episodes_lengths

#Stochastic Sarsa Agent         
class Stoch_SARSAAgent:
    def __init__(self, args:Args):
        self.args=args
        self.gamma=0.95
        self.env = gym.make(args.env_id)
        self.C=self.args.C
        
        # set-up
        if args.env_id=="FrozenLake-v1":
            self.q_table=np.full((self.env.observation_space.n, self.env.action_space.n),0.9)
        else:
            self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.states_history={}
        self.D={}
        self.Z={}
        self.alphas=[]
        self.epsilons=[]
        # results
        self.episodes_lengths=[]
        self.rewards=[]
        self.sum_rewards=0
        self.accumulative_rewards=[]

    def select_action(self,obs):
        self.D,epsilon=update_D(obs,self.D)
        if np.random.rand() <= epsilon:
            return random.randint(0, self.env.action_space.n - 1)
        
        Action_subset=Subset_function(self.env.action_space.n,self.states_history[obs],self.C)
        return Action_subset[np.argmax(self.q_table[obs,Action_subset]) ]
    
    def learn(self, obs, action, reward, next_obs,next_action, done):
        self.Z,alpha=update_Z(obs,action,self.Z)
        target = reward
        if not done:
            target += self.gamma * self.q_table[next_obs, next_action]
        
        self.q_table[obs, action] += alpha * (target - self.q_table[obs, action])   
         
    def train(self):
        obs, _ = self.env.reset()
        if obs not in self.states_history:
            self.states_history[obs] = deque(maxlen=self.args.memory_size)
            
        action=self.select_action(obs)
        for self.global_step in range(self.args.total_timesteps):
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.states_history[obs].append(action)
            done = terminated or truncated
            
            if next_obs not in self.states_history:
                self.states_history[next_obs] = deque(maxlen=self.args.memory_size)
            next_action=self.select_action(next_obs)
            self.learn(obs,action,reward,next_obs,next_action,done)
            obs=next_obs
            action=next_action
            if done:
                self.episodes_lengths.append(self.global_step)
                obs, _ = self.env.reset()
                
            # results
            self.rewards.append(reward)
            self.sum_rewards+=reward
            self.accumulative_rewards.append(self.sum_rewards)
        return self.rewards,self.accumulative_rewards,self.episodes_lengths
 
# DQN Agent
class DQNAgent:
    def __init__(self, args: Args, stoch=False, double=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.args = args
        self.stoch = stoch
        self.double = double
        self.env = gym.make_vec(self.args.env_id,self.args.num_envs)
        
        self.actions_list=discretize_action_space(self.env,self.args.i)
        self.actions_tensor = torch.tensor(self.actions_list, dtype=torch.float32).to(self.device)
        
        self.q_network = QNetwork(self.env).to(self.device)
        self.target_network = QNetwork(self.env).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
        self.epsilons = epsilon_fun()
        
        self.log2_actions = round(np.log2(len(self.actions_list))) 
        self.batch_size = 2 * self.log2_actions
        self.buffer_size = 2 * self.batch_size
        
        self.replay_buffer = ReplayBuffer(
            self.buffer_size,
            self.env.single_observation_space,
            self.env.single_action_space,
            self.device,
            handle_timeout_termination=False)
        
        wandb.init(
        project="Stochastic_QLearning_DQN",
        mode="offline",

        config={"environment": self.args.env_id,
                "num_env": self.args.num_envs,
                "buffer size": self.buffer_size,
                "batch size": self.batch_size,
                "steps": self.args.total_timesteps,
                "learning starts": self.args.learning_starts,
                "learning frequency": self.args.train_frequency,
                "target frequency": self.args.target_network_frequency
                })
                
        self.average_rewards=[]
        self.lengths=np.zeros(self.args.num_envs)
        self.sum_reward = np.zeros(self.args.num_envs)
        #self.random_actions=[]

    def select_action(self, obs, Random_actions):
        wandb.log({'epsilon': self.epsilons[self.global_step]})
        
        if random.random() < self.epsilons[self.global_step] :
            action_indices = np.random.choice(self.actions_list.shape[0], size=self.args.num_envs, replace=True)
            action = self.actions_list[action_indices]
            
            for i in range(self.args.num_envs):
                wandb.log({"actions random": action_indices[i]})
                
            return action 
    
        else:
            
            if self.stoch:
                #if args.env_id=="Breakout-v4" or args.env_id == "Acrobot-v1" or args.env_id == "LunarLander-v2":
                    #obs=obs.reshape(obs.shape[0],-1)
                
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                
                #sample batch_size elements from replay buffer
                data= self.replay_buffer.sample(self.batch_size)
                actions_set = np.concatenate((data.actions.cpu().numpy(), Random_actions), axis=0)
                actions_set = torch.tensor(actions_set,dtype=torch.float)
            
                action,indices=max_action(obs_tensor,actions_set,self.q_network)
                
            else:
                
                #if args.env_id=="Breakout-v4" or args.env_id == "Acrobot-v1" or args.env_id == "LunarLander-v2":
                    #obs=obs.reshape(obs.shape[0],-1)
                    #actions_tensor=self.actions_tensor.reshape(self.actions_tensor.shape[0],1)
                #else:
                actions_tensor=self.actions_tensor
                    
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                
                action,indices=max_action(obs_tensor,actions_tensor,self.q_network)
                
           
            for i in range(self.args.num_envs):
                wandb.log({"actions selected": indices[i]})
                
            return action
            
    def train(self):
        self.episode_step = 0
        obs, _ = self.env.reset()
        
        for self.global_step in range(self.args.total_timesteps):
            
            if self.stoch:
                random_action_indices = np.random.choice(self.actions_list.shape[0], size= self.log2_actions, replace=False) 
                random_actions = self.actions_list[random_action_indices]
                
            else:
                random_actions = None
                
            action = self.select_action(obs,random_actions)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = np.logical_or(terminated,truncated)
            
            for i in range(self.args.num_envs):
                self.replay_buffer.add(obs[i], next_obs[i], action[i], reward[i], done[i],_)
                wandb.log({"actions": action[i]})
                wandb.log({"instant reward": reward[i]})
                
            obs = next_obs
            
            # LEARNING
            if self.global_step > self.args.learning_starts:
                if self.global_step % self.args.train_frequency == 0:
                    self.update_q_network()

                # Update target network
                if self.global_step % self.args.target_network_frequency == 0:
                    self.update_target_network()
                    
            self.sum_reward = self.sum_reward + reward
            self.lengths = self.lengths + np.ones(self.args.num_envs)
            
            if done.any():
                for i in range(self.args.num_envs): 
                    if done[i]:
                        self.episode_step +=1
                        wandb.log({"reward_per_episode": self.sum_reward[i] })
                        wandb.log({"episode_length": self.lengths[i]})
                        wandb.log({"number of episodes": self.episode_step})
                        self.average_rewards.append(self.sum_reward[i])
                        self.lengths[i] = 0
                        self.sum_reward[i] = 0
            if self.episode_step >= self.args.episode_numbers:
                return self.average_rewards
                break
        return self.average_rewards
    def test_max_action(self,loop):
        obs,_=self.env.reset()
        if self.stoch:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            
            random_action_indices = np.random.choice(self.actions_list.shape[0], size= self.log2_actions + self.batch_size, replace=False) 
            random_actions = self.actions_list[random_action_indices]
    
            actions_set = torch.tensor(random_actions,dtype=torch.float)
            start_time=time.time()
            for i in range(loop):
                action,indices=max_action(obs_tensor,actions_set,self.q_network)
                wandb.log({"time": time.time()-start_time})
            end_time=time.time()
            return end_time-start_time
        else:
            random_actions = None
            actions_tensor=self.actions_tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            start_time=time.time()
            for i in range(loop):
                action,indices=max_action(obs_tensor,actions_tensor,self.q_network)
                wandb.log({"time": time.time()-start_time})
            end_time=time.time()
            
            return end_time-start_time
            
    def update_q_network(self):
        data = self.replay_buffer.sample(self.batch_size)
        rewards=data.rewards.to(self.device)
        
        if self.stoch:
            with torch.no_grad():
                if self.double:
                    target_values = Target_Values(data.next_observations, data.actions,rewards,data.dones,self.target_network,self.q_network,self.args.gamma)
                else:
                    target_values = Target_Values(data.next_observations, data.actions,rewards,data.dones,self.target_network,self.target_network,self.args.gamma)        
        else:
            with torch.no_grad():
                if self.double: 
                    target_values = Target_Values(data.next_observations, self.actions_tensor,rewards,data.dones,self.target_network,self.q_network,self.args.gamma)
                else:
                    target_values = Target_Values(data.next_observations, self.actions_tensor,rewards,data.dones,self.target_network,self.target_network,self.args.gamma)
        
        old_val = self.q_network(torch.cat((data.observations.float(), data.actions.float()), dim=-1))
        
        for i in range(len(target_values)):
            wandb.log({'predicted values': old_val[i]})
            wandb.log({'target values': target_values[i]})
        loss = F.mse_loss(old_val, target_values) 
        wandb.log({'loss':loss})
        
        clipped_loss = torch.clamp(loss, min=-1, max=1)
        wandb.log({'clipped loss':clipped_loss})
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        q_state_dict = self.q_network.state_dict()
        target_state_dict = self.target_network.state_dict()

        for name in target_state_dict.keys():
            target_state_dict[name] = (
                self.args.tau * q_state_dict[name] + (1 - self.args.tau) * target_state_dict[name]
                )
        self.target_network.load_state_dict(target_state_dict)