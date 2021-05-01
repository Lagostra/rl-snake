import math
import random

import gym
from gym.utils import seeding

HEIGHT = 21
WIDTH = 21

SNAKE_START_LOC_X = 10
SNAKE_START_LOC_Y = 10

REWARD_APPLE = 100
REWARD_BODY_COLLISION = -50
REWARD_WALL_COLLISION = -50
REWARD_CLOSER_TO_APPLE = 40
REWARD_FARTHER_FROM_APPLE = -1


def clamp(x, y):
    return min(WIDTH - 1, max(0, x)), min(HEIGHT - 1, max(0, y))


class HeadlessSnake(gym.Env):

    def __init__(self):
        super(HeadlessSnake, self).__init__()

        self.width = WIDTH
        self.height = HEIGHT

        self.done = False
        self.seed()
        self.reward = 0
        self.action_space = 4

        self.total, self.maximum = 0, 0

        self.snake_head = Entity()
        self.snake_body = []
        self.add_to_body()

        self.apple = Entity()
        self.move_apple(first=True)

        self.distance_to_apple = 0
        self.distance_to_apple = 0
        self.measure_distance_to_apple()
        self.prev_distance_to_apple = math.inf

        self.state_space = len(self.get_state())

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def random_coordinates(self):
        x = random.randint(0, WIDTH - 1)
        y = random.randint(0, HEIGHT - 1)
        return x, y

    def move_snake(self):
        if self.snake_head.direction == 0:  # Up
            self.snake_head.y -= 1
        elif self.snake_head.direction == 1:  # Right
            self.snake_head.x += 1
        elif self.snake_head.direction == 2:  # Down
            self.snake_head.y += 1
        elif self.snake_head.direction == 3:  # Left
            self.snake_head.x -= 1
        else:
            self.reward = 0

    def go_up(self):
        if self.snake_head.direction != 2:
            self.snake_head.direction = 0

    def go_down(self):
        if self.snake_head.direction != 0:
            self.snake_head.direction = 2

    def go_right(self):
        if self.snake_head.direction != 3:
            self.snake_head.direction = 1

    def go_left(self):
        if self.snake_head.direction != 1:
            self.snake_head.direction = 3

    def move_apple(self, first=False):
        if first or self.snake_head.x == self.apple.x and self.snake_head.y == self.apple.y:
            while True:
                self.apple.x, self.apple.y = self.random_coordinates()
                if not self.body_check_apple():
                    break
            if not first:
                self.update_score()
                self.add_to_body()
            return True

    def update_score(self):
        self.total += 1
        self.maximum = max(self.total, self.maximum)

    def update_episode(self, episode):
        pass

    def reset_score(self):
        self.total = 0

    def add_to_body(self):
        body = Entity()
        self.snake_body.append(body)

    def move_snake_body(self):
        if len(self.snake_body) > 0:
            for index in range(len(self.snake_body) - 1, 0, -1):
                x = self.snake_body[index - 1].x
                y = self.snake_body[index - 1].y
                self.snake_body[index].x = x
                self.snake_body[index].y = y

            self.snake_body[0].x = self.snake_head.x
            self.snake_body[0].y = self.snake_head.y

    def measure_distance_to_apple(self):
        self.prev_distance_to_apple = self.distance_to_apple
        self.distance_to_apple = math.sqrt((self.snake_head.x - self.apple.x) ** 2
                                           + (self.snake_head.y - self.apple.y) ** 2)

    def body_check_snake(self):
        if len(self.snake_body) > 1:
            for body in self.snake_body[1:]:
                if body.x == self.snake_head.x and body.y == self.snake_head.y:
                    self.reset_score()
                    return True
        return False

    def body_check_apple(self):
        if len(self.snake_body) > 1:
            for body in self.snake_body:
                if body.x == self.apple.x and body.y == self.apple.y:
                    return True
        return False

    def wall_check(self):
        if self.snake_head.x < 0 or self.snake_head.x >= WIDTH or self.snake_head.y < 0 or self.snake_head.y >= HEIGHT:
            self.reset_score()
            return True
        return False

    def reset(self):
        self.snake_body = []
        self.snake_head.x, self.snake_head.y = SNAKE_START_LOC_X, SNAKE_START_LOC_Y
        self.add_to_body()
        self.snake_head.direction = -1
        self.reward = 0
        self.total = 0
        self.done = False
        self.move_apple(first=True)

        state = self.get_state()

        return state

    def bye(self):
        pass

    def step(self, action):
        if action == 0:
            self.go_up()
        elif action == 1:
            self.go_right()
        elif action == 2:
            self.go_down()
        elif action == 3:
            self.go_left()

        self.calculate_reward()
        state = self.get_state()
        return state, self.reward, self.done, {}

    def calculate_reward(self):
        reward_given = False
        self.move_snake()

        if self.move_apple():
            self.reward == REWARD_APPLE
            reward_given = True

        self.move_snake_body()
        self.measure_distance_to_apple()

        if self.body_check_snake():
            self.reward = REWARD_BODY_COLLISION
            reward_given = True
            self.done = True

        if self.wall_check():
            self.reward = REWARD_WALL_COLLISION
            reward_given = True
            self.done = True

        if not reward_given:
            if self.distance_to_apple < self.prev_distance_to_apple:
                self.reward = REWARD_CLOSER_TO_APPLE
            else:
                self.reward = REWARD_FARTHER_FROM_APPLE

    def get_2d_game_world(self):
        snake_x, snake_y = clamp(self.snake_head.x, self.snake_head.y)

        map = [[[0] * 3 for _ in range(WIDTH)] for _ in range(HEIGHT)]
        map[snake_y][snake_x][0] = 1
        for b in self.snake_body:
            b_x, b_y = clamp(b.x, b.y)
            map[b_x][b_x][1] = 1

        apple_x, apple_y = clamp(self.apple.x, self.apple.y)
        map[apple_y][apple_x][2] = 1

        return map

    def get_heuristic_state(self):
        if self.snake_head.y == 1:
            wall_up, wall_down = 1, 0
        elif self.snake_head.y == HEIGHT - 1:
            wall_up, wall_down = 0, 1
        else:
            wall_up, wall_down = 0, 0

        if self.snake_head.x == 1:
            wall_left, wall_right = 1, 0
        elif self.snake_head.x == WIDTH - 1:
            wall_left, wall_right = 0, 1
        else:
            wall_right, wall_left = 0, 0

        # body close
        body_up = False
        body_right = False
        body_down = False
        body_left = False

        if len(self.snake_body) > 3:
            for body in self.snake_body[3:]:
                if abs(body.x - self.snake_head.x) + abs(body.y - self.snake_head.y) == 1:
                    if body.y < self.snake_head.y:
                        body_up = True
                    elif body.y > self.snake_head.y:
                        body_down = True
                    elif body.x < self.snake_head.x:
                        body_left = True
                    else:
                        body_right = True

        '''
        state = [
            int(wall_up or body_up),
            int(wall_right or body_right),
            int(wall_down or body_down),
            int(wall_left or body_left)
        ]
        '''

        state = [
            self.snake_head.x,
            self.snake_head.y,
            self.snake_head.direction,
            self.snake_head.x - self.apple.x,
            self.snake_head.y - self.apple.y,
            int(wall_up or body_up),
            int(wall_right or body_right),
            int(wall_down or body_down),
            int(wall_left or body_left)
        ]

        return state

    def get_state(self):
        return self.get_2d_game_world()


class Entity:

    def __init__(self):
        self.x = 0
        self.y = 0

        # 0 = up, 1 = right, 2 = down, 3 = left
        self.direction = 0