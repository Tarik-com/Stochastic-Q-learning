import torch.nn as nn
import torch
import numpy as np
from Args import *
args=Args_()

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
                nn.Linear(env.observation_space.shape[1]+env.action_space.shape[1], 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1))

    def forward(self, x):
        return self.network(x)
    
        
