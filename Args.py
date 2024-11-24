import os
from dataclasses import dataclass

@dataclass
class Args_:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str ="InvertedPendulum-v4"#"Acrobot-v1"#"MountainCarContinuous-v0"#"Breakout-v4"#"HalfCheetah-v4"#"InvertedPendulum-v4"#"FrozenLake-v1"#"CliffWalking-v0"##"CliffWalking-v0"#"HalfCheetah-v4"#"InvertedPendulum-v4"#"FrozenLake-v1"## # "InvertedPendulum-v4" # # # #"FrozenLake-v1" #
    """the id of the environment"""
    i: int = None 
    total_timesteps: int =1_875_000#6_250_000 # 1_875_000 #
    """total timesteps of the experiments"""
    num_envs: int = 16
    """the number of parallel game environments"""
    buffer_size: int = 100_000
    """the replay memory buffer size"""
    learning_rate: float = 0.001
    """the learning rate of the optimizer"""
    gamma: float = 0.95
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    max_epsilon: float = 1
    """the starting epsilon for exploration"""
    min_epsilon: float = 0.01
    """the ending epsilon for exploration"""
    epsilon_decay_rate: float = 0.995
    """the epsilon decay rate"""
    learning_starts: int = int(50_000 /num_envs)
    """timestep to start learning"""
    train_frequency: int = 1
    """the frequency of training"""
    target_network_frequency: int = int(5_000/num_envs) 
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
            
            
            
            
 

