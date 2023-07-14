import importlib
import os
import random
from collections import deque

import numpy as np
import torch

from game import Direction, Point, BLOCK_SIZE
from qtrainer import QTrainer

MAX_MEMORY = 100_000


class Agent:

    def __init__(self, mode='run', ai_details=None):
        if ai_details is None:
            ai_details = {}

        self.ai_details = ai_details

        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.batch_size = ai_details.get('batch_size')

        self.epsilon = ai_details.get('epsilon')  # randomness
        self.gamma = ai_details.get('discount_rate')  # discount rate
        self.name = ai_details.get('name')

        module = importlib.import_module('model.' + self.name)
        model_class = getattr(module, 'LinearQNet')

        self.model = model_class(3)

        self.mode = mode
        if mode == 'train':
            self.trainer = QTrainer(
                self.model,
                lr=ai_details.get('learning_rate'),
                gamma=ai_details.get('discount_rate'))
        elif mode == 'run':
            self.load_model(self.name + '.pth')

        self.n_games = 0
        self.score = 0
        self.record = 0
        self.game_over = False
        self.head = None
        self.snake = []
        self.direction = Direction.RIGHT


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached


    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        final_move = [0, 0, 0]

        self.epsilon = 100 - self.n_games / 10
        if random.randint(0, 200) < self.epsilon and self.mode == 'train':
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


    def move(self, game):
        if self.game_over:
            return

        state_old = self.model.get_state(self, game)

        final_move = self.get_action(state_old)

        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(final_move, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(final_move, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

        self.snake.insert(0, self.head)

        reward = 0

        if game.is_collision(self.head) or game.frame_iteration > 100 * len(self.snake):
            self.game_over = True
            reward = -10
            self.snake = []
        else:
            if self.head == game.food:
                self.score += 1
                reward = 10
                game.place_food()
            else:
                self.snake.pop()

        if self.mode == 'train':
            state_new = self.model.get_state(self, game)

            self.train_short_memory(state_old, final_move, reward, state_new, self.game_over)

            self.remember(state_old, final_move, reward, state_new, self.game_over)

        if self.game_over:
            self.n_games += 1
            if self.mode == 'train':
                self.train_long_memory()

            if self.score > self.record:
                self.record = self.score
                if self.mode == 'train':
                    self.save_model(self.name + '.pth')


    def save_model(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.model.state_dict(), file_name)


    def load_model(self, file_name='model.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        self.model.load_state_dict(torch.load(file_name))
        self.model.eval()
