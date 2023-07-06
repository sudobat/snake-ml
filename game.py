import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 255, 100)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 80

class SnakeGameAI:

    def __init__(self, w=640, h=480, agents=None):
        if agents is None:
            agents = {}
        self.w = w
        self.h = h
        self.agents = agents
        self.food = None
        self.frame_iteration = 0
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        i = -2 * BLOCK_SIZE
        for agent in self.agents.values():
            # init game state
            agent.direction = Direction.RIGHT

            agent.head = Point(self.w/2, self.h/2 + i)
            agent.snake = [agent.head,
                          Point(agent.head.x-BLOCK_SIZE, agent.head.y),
                          Point(agent.head.x-(2*BLOCK_SIZE), agent.head.y)]

            agent.game_over = False
            agent.score = 0

            i += BLOCK_SIZE

        self.food = None
        self.frame_iteration = 0
        self.place_food()


    def play_step(self):
        self.frame_iteration += 1

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        for agent in self.agents.values():
            # 2. move
            agent.move(self)
            # agent.snake.insert(0, agent.head)

            # 3. check if game over
            # agent.reward = 0
            # if self.is_collision(agent.head) or self.frame_iteration > 100 * len(agent.snake):
            #     agent.game_over = True
            #     agent.reward = -10
            #     return agent.reward, agent.game_over, agent.score

            # 4. place new food or just move
            # if agent.head == self.food:
            #     agent.score += 1
            #     agent.reward = 10
            #     self._place_food()
            # else:
            #     agent.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        # return agent.reward, agent.game_over, agent.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.agents[0].head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself or other snakes
        for agent in self.agents.values():
            if pt in agent.snake[1:]:
                return True

        return False


    def place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)

        for agent in self.agents.values():
            if self.food in agent.snake:
                self.place_food()


    def _update_ui(self):
        self.display.fill(BLACK)

        i = 0
        for agent in self.agents.values():
            COLOR1 = BLUE1
            COLOR2 = BLUE2

            if i == 1:
                COLOR1 = GREEN1
                COLOR2 = GREEN2

            for pt in agent.snake:
                pygame.draw.rect(self.display, COLOR1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, COLOR2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

            i += 1

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        i = 0
        for agent in self.agents.values():
            text = font.render(agent.name + " - Score: " + str(agent.score) + " - Record: " + str(agent.record), True, WHITE)
            self.display.blit(text, [0, i])
            i += 2*BLOCK_SIZE

        pygame.display.flip()


    # def _move(self, action):
    #     # [straight, right, left]
    #
    #     clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
    #     idx = clock_wise.index(self.direction)
    #
    #     if np.array_equal(action, [1, 0, 0]):
    #         new_dir = clock_wise[idx] # no change
    #     elif np.array_equal(action, [0, 1, 0]):
    #         next_idx = (idx + 1) % 4
    #         new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
    #     else: # [0, 0, 1]
    #         next_idx = (idx - 1) % 4
    #         new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d
    #
    #     self.direction = new_dir
    #
    #     x = self.head.x
    #     y = self.head.y
    #     if self.direction == Direction.RIGHT:
    #         x += BLOCK_SIZE
    #     elif self.direction == Direction.LEFT:
    #         x -= BLOCK_SIZE
    #     elif self.direction == Direction.DOWN:
    #         y += BLOCK_SIZE
    #     elif self.direction == Direction.UP:
    #         y -= BLOCK_SIZE
    #
    #     self.head = Point(x, y)