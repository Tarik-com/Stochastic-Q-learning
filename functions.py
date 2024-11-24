
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
#%matplotlib inline
import itertools
from collections import deque
import importlib
import Args
importlib.reload(Args)
from  Args import *
args=Args_()


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def Subset_function(action_size,Memory_subset,C):
    """return the subset of action that contains the memory subset and a random selection of action"""
        
    tmp= np.setdiff1d(np.arange(action_size),Memory_subset)
    Random_subset=np.random.choice(tmp,(C-len(Memory_subset)),replace=False)
    Action_subset = np.concatenate((np.array(Memory_subset), Random_subset))
    return Action_subset.astype('int')
    
def update_D(obs,D):
    """return the number of time a state was visited and the epsilon for that state"""
    if type(obs)==int:
        state_key=obs
    else:
        state_key = tuple(obs)
    D[state_key]=D.get(state_key,0)+1
    epsilon=1/np.sqrt(D.get(state_key))
    return D,epsilon

def update_Z(obs,action,Z):
    """return the number of time a pair state action has been visited and return the learning rate alpha"""
    if type(obs) == int:
        state_action_key=(obs,action)
    else:
        state_action_key=(tuple(obs),action)
    Z[state_action_key]=Z.get(state_action_key,0)+1
    alpha=1/(Z.get(state_action_key))**0.8
    #print(f"value: {Z.get(state_action_key)} **0.8 {(Z.get(state_action_key))**0.8} alpha 1/... {alpha} ")
    return Z,alpha
 
def epi_length_convert(data):
    result=[]
    for i in range(len(data[:-1])):
        result.append(data[i+1]-data[i])
    return result
        

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_reward(rewards,window_size,name):
    plt.figure(figsize=(14,8))
    #plt.plot(rewards)
    average_rewards = moving_average(rewards, window_size)
    plt.plot(np.arange(window_size - 1, len(rewards)), average_rewards, label='Moving Average', color='red')
    plt.ylabel('Total Reward')
    plt.xlabel('Episode')
    plt.title(f"Reward per episode for {name}")
    plt.show()

def subplots_(data,window_size,names,y_name,title_,hline=True):
    n=len(data)
    
    fig,axs=plt.subplots(n,1,figsize=(15,5*n),sharex=True)
    for i, (key, values) in enumerate(data.items()):
        tmp=moving_average(values,window_size)
        axs[i].plot(np.arange(window_size - 1, len(values)), tmp, label='Moving Average', color='red')
        axs[i].plot(values,label=f"episodes: {len(values)}")
        if hline:
            axs[i].axhline(y=200, color='r', linestyle='--')
        axs[i].set_ylabel(y_name)
        axs[i].set_xlabel('Episode')
        axs[i].set_title(f"{title_} {names[i]}")
        axs[i].legend()
    plt.legend()
    plt.tight_layout()
    plt.show()

def one_plot(data,window_size,names,y_name,title_,hline=True,all_=False):
    
    plt.figure(figsize=(14,8))
    for i, (key, values) in enumerate(data.items()):
        tmp=moving_average(values,window_size)
        plt.plot(np.arange(window_size - 1, len(values)), tmp, label=names[i])
        if all_:
            plt.plot(values)
    if hline:
        plt.axhline(y=200, color='r', linestyle='--',label="target")
    plt.ylabel(y_name)
    plt.xlabel('Episode')
    plt.title(f"{title_}")
    plt.tight_layout()
    plt.legend()
    plt.show()

def format_func(value, tick_position):
    return f'{int(value):_}'

def one_plot_1(data:list,names:list,window_size=100,y_name="Rewards",x_name="Steps",title=""):
    plt.figure(figsize=(7,5))
    i=0
    arr = np.array(data)
    if arr.ndim==1:
        tmp=moving_average(data,window_size)
        plt.plot(np.arange(window_size - 1, len(data)), tmp,label=names)
    else:
        for d in data:
            tmp=moving_average(d,window_size)
            plt.plot(np.arange(window_size - 1, len(d)), tmp,label=f"{names[i]}")
            
            i+=1
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
    #plt.gca().yaxis.set_major_formatter(FuncFormatter(format_func))
    
    plt.ylabel(y_name)
    plt.xlabel(x_name)
    plt.title(title)
    plt.tight_layout()
    plt.legend()
    plt.show()


def discretize_action_space(env,i):
    if args.env_id=="Breakout-v4" or args.env_id == "Acrobot-v1":
        n=env.single_action_space.n
        return np.linspace(0, n-1,n,dtype=int)
    else:
        d=env.action_space.shape[1]
        low_bound=env.action_space.low[0][0]
        high_bound=env.action_space.high[0][0]
        a=np.linspace(low_bound,high_bound,i)

        return np.array(list(itertools.product(a,repeat=d)))

def epsilon_fun():
    
    epsilon = args.max_epsilon
    epsilon_list=[]
    # Calculate epsilon for each step
    for step in range(args.total_timesteps):
        epsilon_list.append(epsilon)
        epsilon = max(epsilon * args.epsilon_decay_rate, args.min_epsilon) 
    return epsilon_list

def max_action(obs_tensor ,actions_tensor,q_network):
    expanded_obs = obs_tensor.unsqueeze(1).expand(-1,actions_tensor.shape[0], -1) #shape: [n,num_actions, obs_dim]
    expanded_obs = expanded_obs.reshape(-1, obs_tensor.shape[1])  # shape: [n * num_actions, obs_dim]

    actions_tensor = torch.tensor(actions_tensor)
    input_tensor = torch.cat((expanded_obs, actions_tensor.repeat(args.num_envs,1)), dim=-1) #shape: [n*num_actions, obs_dim + action_dim]
    
    with torch.no_grad():
        q_values = q_network(input_tensor) #shape [n*num_actions,1]
    q_values = q_values.view(args.num_envs, actions_tensor.shape[0], -1) #shape [n,num_actions,1]
    
    best_action_index = torch.argmax(q_values,dim=1).squeeze(1) # [n]
    
    if args.env_id=="Breakout-v4" or args.env_id == "Acrobot-v1":
        action = actions_tensor[best_action_index].reshape(-1).cpu().numpy().astype(int) # [n,action_dim]
    else:
        action = actions_tensor[best_action_index].cpu().numpy() # [n,action_dim]
    return action


def Target_Values(next_obs,actions,rewards,target_network,q_network,gamma):
        """
        The size of next_obs needs to be [batch,obs_dim]
        The size of actions needs to be [num_actions, action_dim]
        Actions is a torch.Tensor
        Observations is torch.Tensor
        """
        expanded_obs = next_obs.float().unsqueeze(1).expand(-1, actions.shape[0], -1)  # [batch, num_actions, obs_dim]
        expanded_actions = actions.unsqueeze(0).expand(next_obs.shape[0], -1, -1)  # [batch, num_actions, action_dim]
        # Combine observations and actions
        obs_actions_combined = torch.cat((expanded_obs, expanded_actions), dim=-1)  # [batch, num_actions, obs_dim + action_dim]
        obs_actions_flattened = obs_actions_combined.view(-1, obs_actions_combined.shape[-1])  # [batch * num_actions, obs_dim + action_dim]
        # Get Q-values
        q_values = target_network(obs_actions_flattened)  # [batch * num_actions, 1]
        q_values = q_values.view(next_obs.shape[0], actions.shape[0])  # [batch, num_actions]
        # Identify best actions
        best_indices = torch.argmax(q_values, dim=1)  # [batch]
        
        #best_action_q_values = q_values[torch.arange(q_values.size(0)), best_indices]  # [batch]
        target_values = rewards + gamma * q_network(torch.cat((next_obs.float(),actions[best_indices]),dim=-1)) # [batch , 1]
        return target_values
        
    
        
        
        
        
        