import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from game import Direction


class LinearQNet(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.linear1 = nn.Linear(4, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return x

    def get_state(self, agent, game):

        dir_l = agent.direction == Direction.LEFT
        dir_r = agent.direction == Direction.RIGHT
        dir_u = agent.direction == Direction.UP
        dir_d = agent.direction == Direction.DOWN

        state = [
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
        ]

        return np.array(state, dtype=int)