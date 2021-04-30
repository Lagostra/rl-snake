import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
from tensorflow.python.keras.saving.save import load_model


class DQN:
    """ Deep Q Network """

    def __init__(self, env, params):

        self.action_space = env.action_space
        self.state_space = env.state_space
        self.gamma = params['gamma']
        self.batch_size = params['batch_size']
        self.epsilon = params['epsilon']
        self.epsilon_min = params['epsilon_min']
        self.epsilon_max = params['epsilon_max']
        self.epsilon_decay = params['epsilon_decay']
        self.learning_rate = params['learning_rate']
        self.memory = deque(maxlen=2500)
        self.model = self.build_model()

    def save_model(self, id, name):
        self.model.save(f'./models/{id}/{name}')

    def load_model(self, id, name):
        self.model = load_model(f'./models/{id}/{name}')

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def build_model(self):
        l = tf.keras.layers
        model = Sequential([
            l.Dense(64, input_shape=self.state_space),
            l.Activation("relu"),
            l.Dense(32),
            l.Activation("relu"),
            l.Dense(self.action_space)
        ])

        model.compile(loss='mse', optimizer=Adam(lr=0.00025))
        return model

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space)

        preds = self.model.predict_on_batch(state)
        return np.argmax(preds)

    def train_with_experience_replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        # Tips:
        # Use self.model.predict_on_batch()
        # Do NOT use much time on this. Ask if you're stuck.
        # The implementation in python can be harder than actually understanding the equation.
        # Find the eqation in the README

        self.model.fit(states, None, epochs=1, verbose=0)

    def update_exploration_strategy(self, episode):
        self.epsilon *= 1 - self.epsilon_decay
