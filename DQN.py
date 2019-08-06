# DQNAgent
# https://github.com/maurock/snake-ga
# https://towardsdatascience.com/how-to-teach-an-ai-to-play-games-deep-reinforcement-learning-28f9b920440a

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard
from tensorflow.summary import FileWriter
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import pdb

class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class DQNAgent:

    def __init__(self, state_size, discount=0.95, replay_mem_size=10000, \
                 minibatch_size=64, epsilon=1, epsilon_min=1e-3, \
                 epsilon_decay=0.9975, learning_rate=1e-3, loss='mse', \
                 optimizer=Adam, hidden_dims=[64,32], \
                 activations=['relu', 'relu', 'linear'], \
                 replay_start_size=None, min_replay_memory_size=1000):
        if len(activations) != len(hidden_dims) + 1:
            raise Exception('The number of activations should be the number of hidden layers + 1')
        self.state_size = state_size
        self.discount = discount
        self.memory = deque(maxlen=replay_mem_size)
        self.minibatch_size = minibatch_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.loss = loss
        self.optimizer = optimizer
        self.hidden_dims = hidden_dims
        self.activations = activations
        if not replay_start_size:
            replay_start_size = replay_mem_size / 2
        self.replay_start_size = replay_start_size
        self.min_replay_memory_size = min_replay_memory_size

        self.model = self.create_model()

    def create_model(self):
        # Input --> ReLU --> ReLU --> Linear --> Output
        layers = [Dense(self.hidden_dims[0], input_dim=self.state_size, activation=self.activations[0])]
        for i in range(1, len(self.activations)-1):
            layers.append(Dense(self.hidden_dims[i], activation=self.activations[i]))
        layers.append(Dense(1, activation=self.activations[-1]))

        model = Sequential(layers)

        # Compile for training
        model.compile(self.optimizer(lr=self.learning_rate), loss=self.loss, metrics=['accuracy'])

        return model

    def update_replay_memory(self, current_state, next_state, reward, done):
        """
        Add the relevant transition information, i.e. current state, reward,
        next state, and done status, to the replay memory.
        """
        self.memory.append([current_state, next_state, reward, done])

    def get_qs(self, state):
        """
        Make a prediction about the score for the given state.
        """
        return self.model.predict(state)[0]

    def best_state(self, states):
        """
        Given a list of states, return the state with the best q-value
        :param states: list of states to look through
        :type states: List[List[int, int, int, int]]
        """
        max_value = None
        best_state = None

        if random.random() <= self.epsilon:
            return random.choice(list(states))
        else:
            for state in states:
                # To get model.predict to like the input
                value = self.get_qs(np.reshape(state, [1, self.state_size]))
                if not max_value or max_value < value:
                    max_value = value
                    best_state = state

        return best_state

    def train(self, minibatch_size=64, epochs=3):
        # grab mini batch of replay memory
        if len(self.memory) < self.min_replay_memory_size:
            # Don't train on a super small replay memory, otherwise we risk
            # training over the same data when sampling to create our
            # minibatches
            return
        # Get minibatch of data
        minibatch = random.sample(self.memory, self.minibatch_size)

        # Obtain the predicted q values for each state given future states
        # transition is a tuple of (state, next_state, reward, done)
        new_states = np.array([transition[1] for transition in minibatch])
        future_qs_list = self.model.predict(new_states)

        X = []
        y = []

        for index, (state, _, reward, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.discount * max_future_q
            else:
                new_q = reward

            X.append(state)
            y.append(new_q)

        self.model.fit(np.array(X), np.array(y), batch_size=self.minibatch_size, verbose=0, shuffle=False)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
