import numpy as np
import random
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import importlib

from collections import deque
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import Network
importlib.reload(Network)
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
        self.env = gym.make(args.env_id)
        self.actions_list=discretize_action_space(self.env,self.args.i)
        self.actions_tensor = torch.tensor(self.actions_list, dtype=torch.float32).to(self.device)
        self.tensor_test=torch.tensor([1,2],dtype=torch.float32)
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
        self.rewards=[]
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
        data = self.replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            expanded_obs = data.observations.float().unsqueeze(1).expand(-1, self.actions_tensor.shape[0], -1) # Shape: [num_observations, num_actions, obs_dim]
            expanded_actions = self.actions_tensor.unsqueeze(0).expand(data.observations.shape[0], -1, -1)# Shape: [num_observations, num_actions, action_dim]
            obs_actions_combined = torch.cat((expanded_obs, expanded_actions), dim=-1) # Shape: [num_observations, num_actions, obs_dim + action_dim]
            obs_actions_flattened = obs_actions_combined.view(-1, obs_actions_combined.shape[-1]) # Shape: [num_observations * num_actions, obs_dim + action_dim]
            q_values = self.target_network(obs_actions_flattened) #shape [num_observations * num_actions, 1]
            q_values = q_values.view(data.observations.shape[0], self.actions_tensor.shape[0]) #shape [num_observations, num_actions]
            best_indices = torch.argmax(q_values, dim=1) 

            target_values= data.rewards +self.args.gamma * self.target_network(torch.cat((data.observations.float(),self.actions_tensor[best_indices]),dim=-1))

        old_val = self.q_network( torch.cat( (data.observations.float(),data.actions.float()),-1 ) ).squeeze()
        loss = F.mse_loss(target_values, old_val)
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
        #self.M_subset=deque(maxlen=args.M) # List
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
        self.global_step = 0
        self.rewards=[]
        self.reward_per_episode=[]
        self.sum_reward=0

    def select_action(self, obs):
        if random.random() <self.epsilons[self.global_step]:
            return np.array(random.choice(self.actions_list))
        else:
            obs_tensor = torch.Tensor(obs).to(self.device)
            Action_subset = self.replay_buffer.sample(self.batch_size).actions
            with torch.no_grad():
                combined_tensors=[torch.cat( (obs_tensor,torch.tensor(a, dtype=torch.float32)) ) for a in Action_subset]
                input_tensor = torch.stack(combined_tensors)
                q_values=self.q_network(input_tensor)
                best_action_index = torch.argmax(q_values)
                action = Action_subset[best_action_index.item()] 
            return np.array(action)

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
        data = self.replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            target_max = self.target_network(data.next_observations)[:,Action_subset].max(dim=1)[0].unsqueeze(1)
            td_target = data.rewards + (self.args.gamma * target_max * (1 - data.dones)) #[batchsize,1]
        
        old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()
        loss = F.mse_loss(td_target, old_val)
        if self.global_step % 100 ==0:
            print(f"loss {loss}")

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
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.env = gym.make(args.env_id)
        self.q_network = QNetwork(self.env).to(self.device)
        self.target_network = QNetwork(self.env).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
        self.replay_buffer = ReplayBuffer(
            args.buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.device,
            handle_timeout_termination=False,
        
        )
        self.writer = SummaryWriter(f"runs/{args.env_id}_{int(time.time())}")
        self.global_step = 0
        self.actions_list=discretize_action_space(self.env,self.args.i)
        self.rewards=[]

    def select_action(self, obs):
        epsilon = linear_schedule(
            self.args.start_e, self.args.end_e, self.args.exploration_fraction * self.args.total_timesteps, self.global_step
        )
        if random.random() < epsilon:
            return np.array(random.choice(self.actions_list))
        else:
            obs_tensor = torch.Tensor(obs).to(self.device)
            with torch.no_grad():
                combined_tensors=[torch.cat( (obs_tensor,torch.tensor(a, dtype=torch.float32)) ) for a in self.actions_list]
                input_tensor = torch.stack(combined_tensors)
                q_values=self.q_network(input_tensor)
                best_action_index = torch.argmax(q_values)
                action = self.actions_list[best_action_index.item()] 
            return np.array(action)

    def train(self):
        obs, _ = self.env.reset(seed=self.args.seed)
        for self.global_step in range(self.args.total_timesteps):
            action = self.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.replay_buffer.add(obs, next_obs, action, reward, done,_)
            obs = next_obs
            self.rewards.append(reward)

            
            # Start learning after a certain number of steps
            if self.global_step > self.args.learning_starts:
                if self.global_step % self.args.train_frequency == 0:
                    self.update_q_network()

                # Update target network
                if self.global_step % self.args.target_network_frequency == 0:
                    self.update_target_network()

            if done:
                obs, _ = self.env.reset(seed=self.args.seed)
        return self.rewards

    def update_q_network(self):
        data = self.replay_buffer.sample(self.args.batch_size)
        with torch.no_grad():
            print(f" max values {self.target_network(torch.cat((data.observations.float(),data.actions.float()),-1)).max(dim=1)[0]}")
            print(f" argmax action {self.target_network(torch.cat((data.observations.float(),data.actions.float()),-1)).argmax(dim=1)}")
            target_max = self.target_network(torch.cat((data.observations.float(),data.actions.float()),-1)).max(dim=1)[0]
            td_target = data.rewards + self.args.gamma * target_max * (1 - data.dones)
        old_val = self.q_network( torch.cat( (data.observations.float(),data.actions.float()),-1 ) )
        loss = F.mse_loss(td_target, old_val)

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
        self.M_subset=deque(maxlen=args.M) # List
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        self.env = gym.make(args.env_id)
        self.q_network = QNetwork(self.env).to(self.device)
        self.target_network = QNetwork(self.env).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
        self.replay_buffer = ReplayBuffer(
            args.buffer_size,
            self.env.observation_space,
            self.env.action_space,
            self.device,
            handle_timeout_termination=False,
        )
        self.writer = SummaryWriter(f"runs/{args.env_id}_{int(time.time())}")
        self.global_step = 0

    def select_action(self, obs):
        epsilon = linear_schedule(
            self.args.start_e, self.args.end_e, self.args.exploration_fraction * self.args.total_timesteps, self.global_step
        )
        if random.random() < epsilon:
            action=self.env.action_space.sample()
            self.M_subset.append(action)
            return action
        else:
            Action_subset=Subset_function(self.env.action_space,self.M_subset)
            obs_tensor = torch.Tensor(obs).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(obs_tensor)
                action=torch.argmax(q_values[Action_subset]).cpu().numpy()
                self.M_subset.append(action)
            return action

    def train(self):
        obs, _ = self.env.reset(seed=self.args.seed)
        for self.global_step in range(self.args.total_timesteps):
            action = self.select_action(obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.replay_buffer.add(obs, next_obs, action, reward, done,_)
            obs = next_obs

            # Start learning after a certain number of steps
            if self.global_step > self.args.learning_starts:
                if self.global_step % self.args.train_frequency == 0:
                    self.update_q_network()

                # Update target network
                if self.global_step % self.args.target_network_frequency == 0:
                    self.update_target_network()

            if done:
                obs, _ = self.env.reset(seed=self.args.seed)

    def update_q_network(self):
        data = self.replay_buffer.sample(self.args.batch_size)
        with torch.no_grad():
            Action_subset=Subset_function(self.env.action_space,self.M_subset)
            target_max = self.target_network(data.next_observations)[:,Action_subset].max(dim=1)[0].unsqueeze(1)
            td_target = data.rewards + (self.args.gamma * target_max * (1 - data.dones)) #[batchsize,1]
        
        old_val = self.q_network(data.observations).gather(1, data.actions).squeeze()
        loss = F.mse_loss(td_target, old_val)
        if self.global_step % 100 ==0:
            print(f"loss {loss}")

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

    

    
    
    
    
    
    
    
    
    