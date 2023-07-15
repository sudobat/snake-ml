import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from game import Direction, SnakeGameAI, BLOCK_SIZE, Point


class LinearQNet(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.linear1 = nn.Linear(11, 650)
        self.linear2 = nn.Linear(650, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def get_state(self, agent, game: SnakeGameAI):
        dir_l = agent.direction == Direction.LEFT
        dir_r = agent.direction == Direction.RIGHT
        dir_u = agent.direction == Direction.UP
        dir_d = agent.direction == Direction.DOWN

        distance_to_food = abs(game.food.x - agent.head.x) + abs(game.food.y - agent.head.y)

        is_collision_l = game.is_collision(Point(agent.head.x - BLOCK_SIZE, agent.head.y))
        is_collision_r = game.is_collision(Point(agent.head.x + BLOCK_SIZE, agent.head.y))
        is_collision_u = game.is_collision(Point(agent.head.x, agent.head.y - BLOCK_SIZE))
        is_collision_d = game.is_collision(Point(agent.head.x, agent.head.y + BLOCK_SIZE))

        is_one_block_from_food = distance_to_food == 1

        is_food_left = game.food.x < agent.head.x
        is_food_right = game.food.x > agent.head.x
        is_food_up = game.food.y > agent.head.y
        is_food_down = game.food.y < agent.head.y


        state = [
            # Danger straight
            (dir_r and is_collision_r) or
            (dir_l and is_collision_l) or
            (dir_u and is_collision_u) or
            (dir_d and is_collision_d),

            # Danger right
            (dir_u and is_collision_r) or
            (dir_d and is_collision_l) or
            (dir_l and is_collision_u) or
            (dir_r and is_collision_d),

            # Danger left
            (dir_d and is_collision_r) or
            (dir_u and is_collision_l) or
            (dir_r and is_collision_u) or
            (dir_l and is_collision_d),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            is_food_left,
            is_food_right,
            is_food_up,
            is_food_down,

            # Food distance


        ]

        return np.array(state, dtype=int)
