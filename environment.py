import turtle
import random
import time
import math
import gym
from gym import spaces
from gym.utils import seeding


HEIGHT = 20      # number of steps vertically from wall to wall of screen
WIDTH = 20       # number of steps horizontally from wall to wall of screen
PIXEL_H = 20*HEIGHT  # pixel height + border on both sides
PIXEL_W = 20*WIDTH   # pixel width + border on both sides

SLEEP = 0.2     # time to wait between steps

GAME_TITLE = 'Snake'
BG_COLOR = 'black'
TEXT_COLOR = 'lime green'

SNAKE_SHAPE = 'square'
SNAKE_COLOR = 'lime green'
SNAKE_START_LOC_H = 0
SNAKE_START_LOC_V = 0

APPLE_SHAPE = 'square'
APPLE_COLOR = 'red'


class Snake(gym.Env):

    def __init__(self, human=False, env_info={'state_space': None}):
        super(Snake, self).__init__()

        self.done = False
        self.seed()
        self.reward = 0
        self.action_space = 4

        self.total, self.maximum = 0, 0
        self.human = human
        self.env_info = env_info

        # GAME CREATION WITH TURTLE (RENDER?)
        # screen/background
        self.screen = turtle.Screen()
        self.screen.title(GAME_TITLE)
        self.screen.bgcolor(BG_COLOR)
        self.screen.tracer(0)
        self.screen.setup(width=PIXEL_W+32, height=PIXEL_H+32)

        # snake
        self.snake = turtle.Turtle()
        self.snake.shape(SNAKE_SHAPE)
        self.snake.speed(0)
        self.snake.penup()
        self.snake.color(SNAKE_COLOR)
        self.snake.goto(SNAKE_START_LOC_H, SNAKE_START_LOC_V)
        self.snake.direction = 'stop'
        # snake body, add first element (for location of snake's head)
        self.snake_body = []
        self.add_to_body()

        # apple
        self.apple = turtle.Turtle()
        self.apple.speed(0)
        self.apple.shape(APPLE_SHAPE)
        self.apple.color(APPLE_COLOR)
        self.apple.penup()
        self.move_apple(first=True)

        # distance between apple and snake
        self.distance_to_apple = math.sqrt((self.snake.xcor()-self.apple.xcor())
                                           ** 2 + (self.snake.ycor()-self.apple.ycor())**2)

        # score
        self.score = turtle.Turtle()
        self.score.speed(0)
        self.score.color(TEXT_COLOR)
        self.score.penup()
        self.score.hideturtle()
        self.score.goto(0, 100)
        self.score.write(f"Total: {self.total}   Highest: {self.maximum}",
                         align='center', font=('Courier', 18, 'normal'))

        self.episode_text = turtle.Turtle()
        self.episode_text.speed(0)
        self.episode_text.color(TEXT_COLOR)
        self.episode_text.penup()
        self.episode_text.hideturtle()
        self.episode_text.goto(-200, 190)
        self.episode_text.write(f"Episode: 1", align='left', font=('Courier', 18, 'normal'))

        # control
        self.screen.listen()
        self.screen.onkey(self.go_up, 'Up')
        self.screen.onkey(self.go_right, 'Right')
        self.screen.onkey(self.go_down, 'Down')
        self.screen.onkey(self.go_left, 'Left')

        # Set state space size
        self.state_space = len(self.get_state())

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def random_coordinates(self):
        apple_x = random.randint(-WIDTH/2, WIDTH/2)
        apple_y = random.randint(-HEIGHT/2, HEIGHT/2)
        return apple_x, apple_y

    def move_snake(self):
        if self.snake.direction == 'stop':
            self.reward = 0
        if self.snake.direction == 'up':
            y = self.snake.ycor()
            self.snake.sety(y + 20)
        if self.snake.direction == 'right':
            x = self.snake.xcor()
            self.snake.setx(x + 20)
        if self.snake.direction == 'down':
            y = self.snake.ycor()
            self.snake.sety(y - 20)
        if self.snake.direction == 'left':
            x = self.snake.xcor()
            self.snake.setx(x - 20)

    def get_snake_direction_code(self):
        return {
            'stop': 0,
            'up': 1,
            'right': 2,
            'down': 3,
            'left': 4
        }[self.snake.direction]

    def go_up(self):
        if self.snake.direction != "down":
            self.snake.direction = "up"

    def go_down(self):
        if self.snake.direction != "up":
            self.snake.direction = "down"

    def go_right(self):
        if self.snake.direction != "left":
            self.snake.direction = "right"

    def go_left(self):
        if self.snake.direction != "right":
            self.snake.direction = "left"

    def move_apple(self, first=False):
        if first or self.snake.distance(self.apple) < 20:
            while True:  # place apple
                self.apple.x, self.apple.y = self.random_coordinates()
                self.apple.goto(round(self.apple.x*20), round(self.apple.y*20))
                if not self.body_check_apple():
                    break
            if not first:
                self.update_score()
                self.add_to_body()
            first = False
            return True

    def update_score(self):
        self.total += 1
        if self.total >= self.maximum:
            self.maximum = self.total
        self.score.clear()
        self.score.write(f"Total: {self.total}   Highest: {self.maximum}",
                         align='center', font=('Courier', 18, 'normal'))

    def update_episode(self, episode):
        self.episode_text.clear()
        self.episode_text.write(f"Episode: {episode}", align='left', font=('Courier', 18, 'normal'))

    def reset_score(self):
        self.score.clear()
        self.total = 0
        self.score.write(f"Total: {self.total}   Highest: {self.maximum}",
                         align='center', font=('Courier', 18, 'normal'))

    def add_to_body(self):
        body = turtle.Turtle()
        body.speed(0)
        body.shape(SNAKE_SHAPE)
        body.color(SNAKE_COLOR)
        body.penup()
        self.snake_body.append(body)

    def move_snakebody(self):
        if len(self.snake_body) > 0:
            for index in range(len(self.snake_body)-1, 0, -1):
                x = self.snake_body[index-1].xcor()
                y = self.snake_body[index-1].ycor()
                self.snake_body[index].goto(x, y)

            self.snake_body[0].goto(self.snake.xcor(), self.snake.ycor())

    def measure_distance_to_apple(self):
        self.prev_distance_to_apple = self.distance_to_apple
        self.distance_to_apple = math.sqrt((self.snake.xcor()-self.apple.xcor())
                                           ** 2 + (self.snake.ycor()-self.apple.ycor())**2)

    def body_check_snake(self):
        if len(self.snake_body) > 1:
            for body in self.snake_body[1:]:
                if body.distance(self.snake) < 20:
                    self.reset_score()
                    return True

    def body_check_apple(self):
        if len(self.snake_body) > 0:
            for body in self.snake_body[:]:
                if body.distance(self.apple) < 20:
                    return True

    def wall_check(self):
        if self.snake.xcor() > 200 or self.snake.xcor() < -200 or self.snake.ycor() > 200 or self.snake.ycor() < -200:
            self.reset_score()
            return True

    def reset(self):
        if self.human:
            time.sleep(1)
        for body in self.snake_body:
            body.goto(1000, 1000)

        self.snake_body = []
        self.snake.goto(SNAKE_START_LOC_H, SNAKE_START_LOC_V)
        self.snake.direction = 'stop'
        self.reward = 0
        self.total = 0
        self.done = False
        self.move_apple(first=True)

        state = self.get_state()

        return state

    def bye(self):
        self.screen.bye()

    def step(self, action):
        if action == 0:
            self.go_up()
        if action == 1:
            self.go_right()
        if action == 2:
            self.go_down()
        if action == 3:
            self.go_left()
        self.calculate_reward()
        state = self.get_state()
        return state, self.reward, self.done, {}

    def calculate_reward(self):
        reward_given = False
        self.screen.update()
        self.move_snake()

        if self.move_apple():  # Fetched apple
            self.reward = 100
            reward_given = True
        self.move_snakebody()
        self.measure_distance_to_apple()

        if self.body_check_snake():  # Body collision
            self.reward = -50
            reward_given = True
            self.done = True
            if self.human:
                self.reset()

        if self.wall_check():  # Wall collission
            self.reward = -50
            reward_given = True
            self.done = True
            if self.human:
                self.reset()

        if not reward_given:  # Check if distance to apple increased or decreased
            if self.distance_to_apple < self.prev_distance_to_apple:
                self.reward = 40
            else:
                self.reward = -1

        if self.human:
            time.sleep(SLEEP)
            state = self.get_state()

    def to_coord(self, position):
        return max(0, min(20, int(position / WIDTH) + 10))

    def get_2d_gameworld(self):
        snake_x = self.to_coord(self.snake.xcor())
        snake_y = self.to_coord(self.snake.ycor())
        map = [[[0 for _ in range(3)] for _ in range(21)] for _ in range(21)]
        map[snake_y][snake_x][0] = 1
        for b in self.snake_body:
            body_x, body_y = self.to_coord(b.xcor()), self.to_coord(b.ycor())
            map[body_y][body_x][1] = 1
        apple_x, apple_y = self.to_coord(self.apple.xcor()), self.to_coord(self.apple.ycor())
        map[apple_y][apple_x][2] = 1

        return map

    def get_heuristic_state(self):
        # snake coordinates abs
        self.snake.x, self.snake.y = self.snake.xcor() / WIDTH, self.snake.ycor() / HEIGHT
        # snake coordinates scaled 0-1
        self.snake.xsc, self.snake.ysc = self.snake.x / WIDTH + 0.5, self.snake.y / HEIGHT + 0.5
        # apple coordintes scaled 0-1
        self.apple.xsc, self.apple.ysc = self.apple.x / WIDTH + 0.5, self.apple.y / HEIGHT + 0.5

        # wall check
        if self.snake.y >= HEIGHT / 2:
            wall_up, wall_down = 1, 0
        elif self.snake.y <= -HEIGHT / 2:
            wall_up, wall_down = 0, 1
        else:
            wall_up, wall_down = 0, 0
        if self.snake.x >= WIDTH / 2:
            wall_right, wall_left = 1, 0
        elif self.snake.x <= -WIDTH / 2:
            wall_right, wall_left = 0, 1
        else:
            wall_right, wall_left = 0, 0

        # body close
        body_up = []
        body_right = []
        body_down = []
        body_left = []
        if len(self.snake_body) > 3:
            for body in self.snake_body[3:]:
                if body.distance(self.snake) == 20:
                    if body.ycor() < self.snake.ycor():
                        body_down.append(1)
                    elif body.ycor() > self.snake.ycor():
                        body_up.append(1)
                    if body.xcor() < self.snake.xcor():
                        body_left.append(1)
                    elif body.xcor() > self.snake.xcor():
                        body_right.append(1)

        if len(body_up) > 0:
            body_up = 1
        else:
            body_up = 0
        if len(body_right) > 0:
            body_right = 1
        else:
            body_right = 0
        if len(body_down) > 0:
            body_down = 1
        else:
            body_down = 0
        if len(body_left) > 0:
            body_left = 1
        else:
            body_left = 0

        '''
        state = [
            int(wall_up or body_up),  # obstacle_up
            int(wall_right or body_right),  # obstacle_right
            int(wall_down or body_down),  # obstacle_down
            int(wall_left or body_left),  # obstacle_left
        ]
        '''

        state = [
            self.snake.xsc,
            self.snake.ysc,
            # self.apple.xsc,
            # self.apple.ysc,
            self.get_snake_direction_code(),
            self.snake.xsc - self.apple.xsc,
            self.snake.ysc - self.apple.ysc,
            int(wall_up or body_up),  # obstacle_up
            int(wall_right or body_right),  # obstacle_right
            int(wall_down or body_down),  # obstacle_down
            int(wall_left or body_left),  # obstacle_left
        ]
        return state

    # TODO: Edit the state space. What is the agent allowed to observe?
    def get_state(self):
        return self.get_2d_gameworld()
