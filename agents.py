import numpy as np
import random
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import importlib

from collections import deque
import buffers
importlib.reload(buffers)
from buffers import *
from torch.utils.tensorboard import SummaryWriter
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
        obs, _ = self.env.reset(seed=self.args.seed)
        for self.global_step in range(self.args.total_timesteps):
            action = self.select_action()
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            obs=next_obs
            if done:
                self.episodes_lengths.append(self.global_step)
                obs, _ = self.env.reset(seed=self.args.seed)
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
        
        self.q_table[obs, action] += 0.05 * (target - self.q_table[obs, action])   
    def train(self):
        obs, _ = self.env.reset(seed=self.args.seed)
        for self.global_step in range(self.args.total_timesteps):
            action = self.select_action(obs)
            
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.learn(obs,action,reward,next_obs,done)
            obs=next_obs
            if done:
                self.episodes_lengths.append(self.global_step)
                obs, _ = self.env.reset(seed=self.args.seed)
            self.rewards.append(reward)
            self.sum_rewards+=reward
            self.accumulative_rewards.append(self.sum_rewards)
        return self.rewards,self.accumulative_rewards,self.episodes_lengths

# Q-learning Stochastic
class Stoch_QLearningAgent:
    def __init__(self, args:Args):
        self.args=args
        self.gamma=0.95
        self.alpha=0.05
        self.env = gym.make(args.env_id)
        self.C=2#int(2*np.log(self.env.action_space.n))
        
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
        
        self.q_table[obs, action] += 0.05 * (target - self.q_table[obs, action])
            
            
    def train(self):
        obs, _ = self.env.reset(seed=self.args.seed)
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
                obs, _ = self.env.reset(seed=self.args.seed)
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
        obs, _ = self.env.reset(seed=self.args.seed)
        for self.global_step in range(self.args.total_timesteps):
            action = self.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.learn(obs,action,reward,next_obs,done)
            obs=next_obs
            if done:
                self.episodes_lengths.append(self.global_step)
                obs, _ = self.env.reset(seed=self.args.seed)
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
        self.C=int(2*np.log(self.env.action_space.n))
        self.R= self.C - self.args.memory_size
        
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
        obs, _ = self.env.reset(seed=self.args.seed)
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
                obs, _ = self.env.reset(seed=self.args.seed)
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
        obs, _ = self.env.reset(seed=self.args.seed)
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
                obs, _ = self.env.reset(seed=self.args.seed)
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
        self.C=int(2*np.log(self.env.action_space.n))
        self.R= self.C - self.args.memory_size
        
        # set-up
        if args.env_id=="FrozenLake-v1":
            self.q_table=np.full((self.env.observation_space.n, self.env.action_space.n),0.9)
        else:
            self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.states_history={}
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
            
            action=random.randint(0, self.env.action_space.n - 1) 
            return action
        Action_subset=Subset_function(self.env.action_space.n,self.states_history[obs],self.C)
        action=np.argmax(self.q_table[obs,Action_subset]) 
        return action
    
    def learn(self, obs, action, reward, next_obs,next_action, done):
        self.Z,alpha=update_Z(obs,action,self.Z)
        target = reward
        if not done:
            target += self.gamma * self.q_table[next_obs, next_action]
        
        self.q_table[obs, action] += alpha * (target - self.q_table[obs, action])   
         
    def train(self):
        obs, _ = self.env.reset(seed=self.args.seed)
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
                obs, _ = self.env.reset(seed=self.args.seed)
                
            # results
            self.rewards.append(reward)
            self.sum_rewards+=reward
            self.accumulative_rewards.append(self.sum_rewards)
        return self.rewards,self.accumulative_rewards,self.episodes_lengths
 
# DQN Agent
class DQNAgent:
    def __init__(self, args: Args):
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.args = args
        ### modification
        self.env = gym.make_vec(args.env_id,self.args.num_envs)
        self.actions_list=discretize_action_space(self.env,self.args.i)
        self.actions_tensor = torch.tensor(self.actions_list, dtype=torch.float32).to(self.device)
        self.q_network = QNetwork(self.env).to(self.device)
        self.target_network = QNetwork(self.env).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
        self.epsilons = epsilon_fun(self.args.total_timesteps)
        self.buffer_size=int(2*np.log(len(self.actions_list)))
        self.batch_size=int(np.log(len(self.actions_list)))
        self.replay_buffer = ReplayBuffer(
            self.buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.device,
            ### modification
            n_envs=self.args.num_envs,
            handle_timeout_termination=False,)
        self.writer = SummaryWriter(f"runs/{args.env_id}_{int(time.time())}")
        
        
        self.reward_per_episode = np.zeros(self.args.num_envs)
        self.sum_reward = np.zeros(self.args.num_envs)

    def select_action(self, obs):
        if random.random() < self.epsilons[self.global_step]:
            return self.actions_list[np.random.choice(self.actions_list.shape[0],size=self.args.num_envs,replace=True)]
        
        else:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)  # [n, obs_dim]
            expanded_obs = obs_tensor.unsqueeze(1).expand(-1,self.actions_tensor.shape[0], -1) #shape: [n,num_actions, obs_dim]
            expanded_obs = expanded_obs.reshape(-1, obs_tensor.shape[1])  # shape: [n * num_actions, obs_dim]
            input_tensor = torch.cat((expanded_obs, self.actions_tensor.repeat(self.args.num_envs,1)), dim=-1) #shape: [n*num_actions, obs_dim + action_dim]
            with torch.no_grad():
                q_values = self.q_network(input_tensor)   
            q_values = q_values.view(self.args.num_envs, self.actions_tensor.shape[0], -1)
            best_action_index = torch.argmax(q_values,dim=1).squeeze(1)
            action = self.actions_tensor[best_action_index].cpu().numpy()
            return action

    def train(self):
        obs, _ = self.env.reset(seed=self.args.seed)
        for self.global_step in range(self.args.total_timesteps):
            action = self.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = np.logical_or(terminated,truncated)
            #print(f"before buffer obs shape= {obs.shape} next obs shape= {next_obs.shape} action shape= {action.shape} reward shape {reward.shape} done shape {done.shape}")
            self.replay_buffer.add(obs, next_obs, action, reward, done,_)
            
            obs = next_obs
            self.sum_reward=self.sum_reward+reward
            
            # Start learning after a certain number of steps
            if self.global_step > self.args.learning_starts:
                if self.global_step % self.args.train_frequency == 0:
                    self.update_q_network()

                # Update target network
                if self.global_step % self.args.target_network_frequency == 0:
                    self.update_target_network()

            if done.any():
                for i in range(self.args.num_envs): 
                    if done[i]:
                        self.reward_per_episode[i] += self.sum_reward[i] 
                        self.sum_reward[i] = 0 
                        obs[i], _ = self.env.reset(seed=self.args.seed) 

        return self.reward_per_episode

    def update_q_network(self):
        data = self.replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            # Expand observations and actions for parallel processing
            expanded_obs = data.observations.float().unsqueeze(2).expand(-1, -1, self.actions_tensor.shape[0], -1)  #[batch,n_envs,num_actions,obs_dim]
            expanded_actions = self.actions_tensor.unsqueeze(0).unsqueeze(0).expand(data.observations.shape[0], data.observations.shape[1], -1, -1) #[batch,n_envs,num_actions,action_dim]
            # Combine observations and actions
            obs_actions_combined = torch.cat((expanded_obs, expanded_actions), dim=-1) #[batch,n_envs,num_actions,obs_dim+action_dim]
            obs_actions_flattened = obs_actions_combined.view(-1, obs_actions_combined.shape[-1]) #[batch*n_envs*num_actions,obs_dim+action_dim]
            # Get Q-values
            q_values = self.target_network(obs_actions_flattened) #[batch*n_envs*num_actions,1]
            q_values = q_values.view(data.observations.shape[0], data.observations.shape[1], self.actions_tensor.shape[0])#[batch,n_envs,num_actions]

            # Identify best actions per environment
            best_indices = torch.argmax(q_values, dim=2) # [batch_size,n_envs]
            best_action_q_values = q_values[torch.arange(data.observations.shape[0]).unsqueeze(1), torch.arange(data.observations.shape[1]), best_indices] # [batch_size,n_envs]

            # Calculate target values
            target_values = data.rewards.squeeze(-1).to(self.device) + self.args.gamma * best_action_q_values # [batch_size,n_envs]

        # Old Q-values and loss calculation
        old_val = self.q_network(torch.cat((data.observations.float(), data.actions.float()), dim=-1))

        loss = F.mse_loss(old_val, target_values.unsqueeze(-1)) #[batch_size, n_envs, 1] == [batch_size, n_envs, 1]

        # Update Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
        if self.global_step % 100 == 0:
            self.writer.add_scalar("losses/td_loss", loss, self.global_step)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def close(self):
        self.env.close()
        self.writer.close()

# Stochastic DQN Agent
class Stoch_DQNAgent:
    def __init__(self, args: Args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.env = gym.make(args.env_id)
        self.actions_list=discretize_action_space(self.env,self.args.i)
        self.q_network = QNetwork(self.env).to(self.device)
        self.target_network = QNetwork(self.env).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
        self.epsilons = epsilon_fun(self.args.total_timesteps)
        self.buffer_size=int(2*np.log(len(self.actions_list)))
        self.batch_size=int(np.log(len(self.actions_list)))
        self.replay_buffer = ReplayBuffer(
            self.buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.device,
            handle_timeout_termination=False,
        )
        self.writer = SummaryWriter(f"runs/{args.env_id}_{int(time.time())}")
        
        #self.global_step = 0
        self.reward_per_episode=[]
        self.sum_reward=0

    def select_action(self, obs):
        if random.random() <self.epsilons[self.global_step] or self.global_step<self.buffer_size:
            return np.array(random.choice(self.actions_list))
        else:
            # sample log(n) actions from buffer of size 2log(n)
            data= self.replay_buffer.sample(self.buffer_size)
            indices = torch.randperm(data.actions.size(0))[:self.batch_size]
            sampled_tensor = data.actions[indices]
            
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, obs_dim]
            expanded_obs = obs_tensor.expand(sampled_tensor.shape[0], -1) #shape: [num_actions, obs_dim]
            input_tensor = torch.cat((expanded_obs, sampled_tensor), dim=-1) #shape: [num_actions, obs_dim + action_dim]
            with torch.no_grad():
                q_values = self.q_network(input_tensor)
            best_action_index = torch.argmax(q_values).item()
            action = sampled_tensor[best_action_index].cpu().numpy()
            return action
        

    def train(self):
        obs, _ = self.env.reset(seed=self.args.seed)
        for self.global_step in range(self.args.total_timesteps):
            action = self.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.replay_buffer.add(obs, next_obs, action, reward, done,_)
            obs = next_obs
            self.sum_reward=self.sum_reward + reward

            # Start learning after a certain number of steps
            if self.global_step > self.args.learning_starts:
                if self.global_step % self.args.train_frequency == 0:
                    self.update_q_network()

                # Update target network
                if self.global_step % self.args.target_network_frequency == 0:
                    self.update_target_network()

            if done:
                self.reward_per_episode.append(self.sum_reward)
                self.sum_reward=0
                obs, _ = self.env.reset(seed=self.args.seed)
        return self.reward_per_episode

    def update_q_network(self):
        data = self.replay_buffer.sample(2*self.batch_size)
        # sample log(n) action from data
        indices = torch.randperm(data.actions.size(0))[:self.batch_size]
        actions_tensor = data.actions[indices]
        with torch.no_grad():
            expanded_obs = data.observations.float().unsqueeze(1).expand(-1, actions_tensor.shape[0], -1) # Shape: [num_observations, num_actions, obs_dim]
            expanded_actions = actions_tensor.unsqueeze(0).expand(data.observations.shape[0], -1, -1)# Shape: [num_observations, num_actions, action_dim]
            
            obs_actions_combined = torch.cat((expanded_obs, expanded_actions), dim=-1) # Shape: [num_observations, num_actions, obs_dim + action_dim]
            obs_actions_flattened = obs_actions_combined.view(-1, obs_actions_combined.shape[-1]) # Shape: [num_observations * num_actions, obs_dim + action_dim]
            
            q_values = self.target_network(obs_actions_flattened) #shape [num_observations * num_actions, 1]
            q_values = q_values.view(data.observations.shape[0], actions_tensor.shape[0]) #shape [num_observations, num_actions]
            
            best_indices = torch.argmax(q_values, dim=1)
            
            best_action_q_values = q_values[range(q_values.size(0)), best_indices]  # Shape: [num_observations]
            target_values = data.rewards.squeeze().to(self.device) + self.args.gamma * best_action_q_values

            
            #target_values= data.rewards.to(self.device) +self.args.gamma * self.target_network(torch.cat((data.observations.float(),actions_tensor[best_indices]),dim=-1))
        
        old_val = self.q_network( torch.cat( (data.observations.float(),data.actions.float()),-1 ) )
        #print(f" old val shape {old_val.shape} target shape {target_values.unsqueeze(1).shape}")
        loss = F.mse_loss(old_val, target_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.global_step % 100 == 0:
            self.writer.add_scalar("losses/td_loss", loss, self.global_step)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def close(self):
        self.env.close()
        self.writer.close()

# DDQN Agent
class DDQNAgent:
    def __init__(self, args: Args):
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.args = args
        self.env = gym.make(args.env_id)
        self.actions_list=discretize_action_space(self.env,self.args.i)
        self.actions_tensor = torch.tensor(self.actions_list, dtype=torch.float32).to(self.device)
        self.q_network = QNetwork(self.env).to(self.device)
        self.target_network = QNetwork(self.env).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
        self.epsilons = epsilon_fun(self.args.total_timesteps)
        self.buffer_size=int(2*np.log(len(self.actions_list)))
        self.batch_size=int(np.log(len(self.actions_list)))
        self.replay_buffer = ReplayBuffer(
            self.buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.device,
            handle_timeout_termination=False,)
        self.writer = SummaryWriter(f"runs/{args.env_id}_{int(time.time())}")
        
        #self.global_step = 0
        self.reward_per_episode=[]
        self.sum_reward=0

    def select_action(self, obs):
        if random.random() < self.epsilons[self.global_step]:
            return np.array(random.choice(self.actions_list))
        else:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, obs_dim]
            expanded_obs = obs_tensor.expand(self.actions_tensor.shape[0], -1) #shape: [num_actions, obs_dim]
            input_tensor = torch.cat((expanded_obs, self.actions_tensor), dim=-1) #shape: [num_actions, obs_dim + action_dim]
            with torch.no_grad():
                q_values = self.q_network(input_tensor)
            best_action_index = torch.argmax(q_values).item()
            action = self.actions_tensor[best_action_index].cpu().numpy()
            return action

    def train(self):
        obs, _ = self.env.reset(seed=self.args.seed)
        for self.global_step in range(self.args.total_timesteps):
            action = self.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.replay_buffer.add(obs, next_obs, action, reward, done,_)
            obs = next_obs
            self.sum_reward=self.sum_reward+reward
            # Start learning after a certain number of steps
            if self.global_step > self.args.learning_starts:
                if self.global_step % self.args.train_frequency == 0:
                    self.update_q_network()

                # Update target network
                if self.global_step % self.args.target_network_frequency == 0:
                    self.update_target_network()

            if done:
                self.reward_per_episode.append(self.sum_reward)
                self.sum_reward=0
                obs, _ = self.env.reset(seed=self.args.seed)
        return self.reward_per_episode

    def update_q_network(self):
        data = self.replay_buffer.sample(self.batch_size)
        
        with torch.no_grad():
            expanded_obs = data.observations.float().unsqueeze(1).expand(-1, self.actions_tensor.shape[0], -1) # Shape: [num_observations, num_actions, obs_dim]
            expanded_actions = self.actions_tensor.unsqueeze(0).expand(data.observations.shape[0], -1, -1)# Shape: [num_observations, num_actions, action_dim]
            
            obs_actions_combined = torch.cat((expanded_obs, expanded_actions), dim=-1) # Shape: [num_observations, num_actions, obs_dim + action_dim]
            obs_actions_flattened = obs_actions_combined.view(-1, obs_actions_combined.shape[-1]) # Shape: [num_observations * num_actions, obs_dim + action_dim]
            
            q_values = self.q_network(obs_actions_flattened) #shape [num_observations * num_actions, 1]
            q_values = q_values.view(data.observations.shape[0], self.actions_tensor.shape[0]) #shape [num_observations, num_actions]
            
            best_indices = torch.argmax(q_values, dim=1)
            
            target_values= data.rewards.to(self.device) +self.args.gamma * self.target_network(torch.cat((data.observations.float(),self.actions_tensor[best_indices]),dim=-1))
    
        old_val = self.q_network( torch.cat( (data.observations.float(),data.actions.float()),-1 ) )

        #print(f" old val shape {old_val.shape} target shape {target_values.shape}")
        loss = F.mse_loss(old_val,target_values)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        if self.global_step % 100 == 0:
            self.writer.add_scalar("losses/td_loss", loss, self.global_step)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def close(self):
        self.env.close()
        self.writer.close()

# Stoch DDQN Agent
class Stoch_DDQNAgent:
    def __init__(self, args: Args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.env = gym.make(args.env_id)
        self.actions_list=discretize_action_space(self.env,self.args.i)
        self.q_network = QNetwork(self.env).to(self.device)
        self.target_network = QNetwork(self.env).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
        self.epsilons = epsilon_fun(self.args.total_timesteps)
        self.buffer_size=int(2*np.log(len(self.actions_list)))
        self.batch_size=int(np.log(len(self.actions_list)))
        self.replay_buffer = ReplayBuffer(
            self.buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.device,
            handle_timeout_termination=False,
        )
        self.writer = SummaryWriter(f"runs/{args.env_id}_{int(time.time())}")
        
        #self.global_step = 0
        self.rewards=[]
        self.reward_per_episode=[]
        self.sum_reward=0

    def select_action(self, obs):
        if random.random() <self.epsilons[self.global_step] or self.global_step<self.buffer_size:
            return np.array(random.choice(self.actions_list))
        else:
            # sample log(n) actions from buffer of size 2log(n)
            data= self.replay_buffer.sample(self.buffer_size)
            indices = torch.randperm(data.actions.size(0))[:self.batch_size]
            sampled_tensor = data.actions[indices]
            
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, obs_dim]
            expanded_obs = obs_tensor.expand(sampled_tensor.shape[0], -1) #shape: [num_actions, obs_dim]
            input_tensor = torch.cat((expanded_obs, sampled_tensor), dim=-1) #shape: [num_actions, obs_dim + action_dim]
            with torch.no_grad():
                q_values = self.q_network(input_tensor)
            best_action_index = torch.argmax(q_values).item()
            action = sampled_tensor[best_action_index].cpu().numpy()
            return action
        

    def train(self):
        obs, _ = self.env.reset(seed=self.args.seed)
        for self.global_step in range(self.args.total_timesteps):
            action = self.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.replay_buffer.add(obs, next_obs, action, reward, done,_)
            obs = next_obs
            self.sum_reward=self.sum_reward+reward
            self.rewards.append(reward)

            # Start learning after a certain number of steps
            if self.global_step > self.args.learning_starts:
                if self.global_step % self.args.train_frequency == 0:
                    self.update_q_network()

                # Update target network
                if self.global_step % self.args.target_network_frequency == 0:
                    self.update_target_network()

            if done:
                self.reward_per_episode.append(self.sum_reward)
                self.sum_reward=0
                obs, _ = self.env.reset(seed=self.args.seed)
        return self.rewards,self.reward_per_episode

    def update_q_network(self):
        data = self.replay_buffer.sample(2*self.batch_size)
        # sample log(n) action from data
        indices = torch.randperm(data.actions.size(0))[:self.batch_size]
        actions_tensor = data.actions[indices]
        with torch.no_grad():
            expanded_obs = data.observations.float().unsqueeze(1).expand(-1, actions_tensor.shape[0], -1) # Shape: [num_observations, num_actions, obs_dim]
            expanded_actions = actions_tensor.unsqueeze(0).expand(data.observations.shape[0], -1, -1)# Shape: [num_observations, num_actions, action_dim]
            
            obs_actions_combined = torch.cat((expanded_obs, expanded_actions), dim=-1) # Shape: [num_observations, num_actions, obs_dim + action_dim]
            obs_actions_flattened = obs_actions_combined.view(-1, obs_actions_combined.shape[-1]) # Shape: [num_observations * num_actions, obs_dim + action_dim]
            
            q_values = self.q_network(obs_actions_flattened) #shape [num_observations * num_actions, 1]
            q_values = q_values.view(data.observations.shape[0], actions_tensor.shape[0]) #shape [num_observations, num_actions]
            
            best_indices = torch.argmax(q_values, dim=1)
            
            target_values= data.rewards.to(self.device) +self.args.gamma * self.target_network(torch.cat((data.observations.float(),actions_tensor[best_indices]),dim=-1))
        
        old_val = self.q_network( torch.cat( (data.observations.float(),data.actions.float()),-1 ) )
        
        #print(f" old val shape {old_val.shape} target shape {target_values.shape}")
        loss = F.mse_loss(old_val,target_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.global_step % 100 == 0:
            self.writer.add_scalar("losses/td_loss", loss, self.global_step)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def close(self):
        self.env.close()
        self.writer.close()

    

    
    
    
    
    
    
    
    
    