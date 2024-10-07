
import numpy as np
import matplotlib.pyplot as plt
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

def one_plot_1(data:list,names:list,window_size,y_name="rewards",x_name="steps",title=""):
    plt.figure(figsize=(7,5))
    i=0
    for d in data:
        tmp=moving_average(d,window_size)
        plt.plot(np.arange(window_size - 1, len(d)), tmp,label=f"{names[i]}")
        i+=1
        
    plt.ylabel(y_name)
    plt.xlabel(x_name)
    plt.title(title)
    plt.tight_layout()
    plt.legend()
    plt.show()


def discretize_action_space(env,i):
    d=env.action_space.shape[0]
    low_bound=env.action_space.low[0]
    high_bound=env.action_space.high[0]
    a=np.linspace(low_bound,high_bound,i)

    return list(itertools.product(a,repeat=d))

def epsilon_fun(n_episodes):
    epsilon_array = np.zeros((n_episodes))
    for i in range(n_episodes):
        epsilon = args.min_epsilon + (args.max_epsilon-args.min_epsilon)*np.exp(-args.epsilon_decay_rate*i)
        epsilon_array[i] = epsilon
    return epsilon_array