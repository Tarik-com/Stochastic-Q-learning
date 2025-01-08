import os
from dataclasses import dataclass

@dataclass
class Args_:

    seed: int = 1
    """seed of the experiment"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""

    # Algorithm specific arguments
    env_id: str ="InvertedPendulum-v4"#"HalfCheetah-v4"#"InvertedPendulum-v4"# #"FrozenLake-v1"#"CliffWalking-v0"
    """the id of the environment"""
    i: int = None 
    total_timesteps: int =30_000 #6_250_000 # 1_875_000 #
    """total timesteps of the experiments"""
    max_step=100
    """steps lkimit per episode"""
    num_envs: int = 1
    """the number of parallel game environments"""

    learning_rate: float = 0.001
    """the learning rate of the optimizer"""
    gamma: float = 0.95
    """the discount factor gamma"""
    tau: float = 0.75
    """the target network update rate"""

    max_epsilon: float = 1
    """the starting epsilon for exploration"""
    min_epsilon: float = 0.01
    """the ending epsilon for exploration"""
    epsilon_decay_rate: float = 0.995
    """the epsilon decay rate"""
    
    learning_starts: int = 50
    """timestep to start learning"""
    train_frequency: int = 1
    """the frequency of training"""
    target_network_frequency: int = 1 
    """the timesteps it takes to update the target network"""
    
    # for the stochastic learning
    
    C:int=2
    """total size of the action subset"""
    
    memory_size: int =2
    """the size of the memory to store tha actions for a state"""
    
    def __post_init__(self):
        if self.env_id == "HalfCheetah-v4":
            self.i = 4
        elif self.env_id == "InvertedPendulum-v4":
            self.i = 512
        elif self.env_id == "MountainCarContinuous-v0":
            self.i = 512
        else:
            self.i=0
            
            
            
            
 

