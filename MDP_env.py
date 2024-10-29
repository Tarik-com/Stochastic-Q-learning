import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MDPEnv(gym.Env):
    def __init__(self):
        super(MDPEnv, self).__init__()
        self.action_space = spaces.Discrete(256)
        self.observation_space = spaces.Discrete(3)
        
        self.state = np.random.choice(3)
        

        self.reward_mean = -50
        self.reward_std = 50

    def reset(self):
        self.state = np.random.choice(3)
        return self.state, {} 

    def step(self, action):

        reward = np.random.normal(self.reward_mean, self.reward_std)
        
        self.state = np.random.choice(3)
        
        # Set a done flag to False (assuming it's an ongoing task)
        done = False

        # Optionally, additional info can be returned
        info = {}
        
        return self.state, reward, done, info

    def render(self):
        # Rendering logic (optional)
        print(f"Current state: {self.state}")

    def close(self):
        # Clean up (optional)
        pass

env=MDPEnv()