import turtle

GAME_TITLE = 'Snake'
BG_COLOR = 'black'
TEXT_COLOR = 'lime green'

SNAKE_SHAPE = 'square'
SNAKE_COLOR = 'lime green'
SNAKE_START_LOC_H = 0
SNAKE_START_LOC_V = 0

APPLE_SHAPE = 'square'
APPLE_COLOR = 'red'

BLOCK_SIZE = 20

class Renderer:

    def __init__(self, environment, episode):
        self.environment = environment

        self.screen = turtle.Screen()
        self.screen.title(GAME_TITLE)
        self.screen.bgcolor(BG_COLOR)
        self.screen.tracer(0)
        self.screen.setup(width=environment.width * BLOCK_SIZE, height=environment.height * BLOCK_SIZE)

        self.apple = turtle.Turtle()
        self.apple.speed(0)
        self.apple.shape(APPLE_SHAPE)
        self.apple.color(APPLE_COLOR)
        self.apple.penup()
        self.apple.goto(environment.apple.x * BLOCK_SIZE, environment.apple.y * BLOCK_SIZE)

        self.score = turtle.Turtle()
        self.score.speed(0)
        self.score.color(TEXT_COLOR)
        self.score.penup()
        self.score.hideturtle()
        self.score.goto(0, 100)

        self.episode_text = turtle.Turtle()
        self.episode_text.speed(0)
        self.episode_text.color(TEXT_COLOR)
        self.episode_text.penup()
        self.episode_text.hideturtle()
        self.episode_text.goto(-200, 190)
        self.episode_text.write(f"Episode: {episode}", align='left', font=('Courier', 18, 'normal'))

        self.snake_body = []

    def update(self):
        self.score.clear()
        self.score.write(f"Total: {self.environment.total}   Highest: {self.environment.maximum}",
                         align='center', font=('Courier', 18, 'normal'))

        self.apple.goto(self.environment.apple.x * BLOCK_SIZE, self.environment.apple.y * BLOCK_SIZE)

        while len(self.snake_body) < len(self.environment.snake_body):
            body = turtle.Turtle()
            body.speed(0)
            body.shape(SNAKE_SHAPE)
            body.color(SNAKE_COLOR)
            self.snake_body.append(body)

        for i in range(len(self.snake_body)):
            screen_body = self.snake_body[i]
            env_body = self.environment.snake_body[i]

            screen_body.goto(env_body.x * BLOCK_SIZE, env_body.y * BLOCK_SIZE)

    def bye(self):
        self.screen.bye()