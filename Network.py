import torch.nn as nn
import numpy as np
from Args import *
args=Args_()

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        if args.env_id=="Breakout-v4":
            self.network = nn.Sequential(
                    nn.Linear(np.prod(env.single_observation_space.shape)+1, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
            )
        else:
            self.network = nn.Sequential(
                    nn.Linear(env.observation_space.shape[1]+env.action_space.shape[1], 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
            )

    def forward(self, x):
        return self.network(x)