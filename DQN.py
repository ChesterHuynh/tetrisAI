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
import pdb
from collections import deque

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

    def __init__(self, state_size, discount=0.98, replay_mem_size=20000, \
                 minibatch_size=512, epsilon=1, \
                 epsilon_stop_episode=1500, epsilon_min=1e-3, \
                 learning_rate=1e-3, loss='mse', \
                 optimizer=Adam, hidden_dims=[64,64], \
                 activations=['relu', 'relu', 'linear'], \
                 replay_start_size=None):
        if len(activations) != len(hidden_dims) + 1:
            raise Exception('The number of activations should be the number of hidden layers + 1')
        self.state_size = state_size
        self.discount = discount
        self.memory = deque(maxlen=replay_mem_size)
        self.minibatch_size = minibatch_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / epsilon_stop_episode
        self.learning_rate = learning_rate
        self.loss = loss
        self.optimizer = optimizer
        self.hidden_dims = hidden_dims
        self.activations = activations
        if not replay_start_size:
            replay_start_size = replay_mem_size / 2
        self.replay_start_size = replay_start_size

        self.model = self.create_model()

    def create_model(self):
        """
        Create neural network architecture.
        """
        # Input --> ReLU --> ReLU --> Linear --> Output
        layers = [Dense(self.hidden_dims[0], input_dim=self.state_size, activation=self.activations[0])]
        for i in range(1, len(self.activations)-1):
            layers.append(Dense(self.hidden_dims[i], activation=self.activations[i]))
        layers.append(Dense(1, activation=self.activations[-1]))

        model = Sequential(layers)

        # Compile for training
        model.compile(self.optimizer(lr=self.learning_rate), loss=self.loss)

        return model

    def update_replay_memory(self, current_state, action, next_state, reward, done):
        """
        Add the relevant transition information, i.e. current state, reward,
        next state, and done status, to the replay memory.
        """
        self.memory.append([current_state, action, next_state, reward, done])

    def get_qs(self, state):
        """
        Make a prediction about the score for the given state.
        We return with the 0th index because predict returns a 2d array
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
                # Reshape so that model.predict likes the input
                value = self.get_qs(np.reshape(state, [1, self.state_size]))
                if not max_value or max_value < value:
                    max_value = value
                    best_state = state

        return best_state

    def train(self, epochs=3):
        """
        Train our neural network to estimate q-values.
        :param minibatch_size: How many samples from our memory do we want to use for training.
        :param epochs: number of epochs to train for.
        :type minibatch_size: int
        :type epochs: int
        """
        # grab mini batch of replay memory
        if len(self.memory) < self.replay_start_size or len(self.memory) < self.minibatch_size:
            # Don't train on a super small replay memory, otherwise we risk
            # training over the same data when sampling to create our
            # minibatches
            return

        # Get minibatch of data
        minibatch = random.sample(self.memory, self.minibatch_size)

        # Obtain the predicted q values for each state given future states
        # Note: transition is a tuple of (state, action, next_state, reward, done)
        next_states = np.array([transition[2] for transition in minibatch])
        next_qs = [x[0] for x in self.model.predict(next_states)]

        X = []
        y = []

        for i, (current_state, action, _, reward, done) in enumerate(minibatch):
            # Update q values according to standard q-learning update rule
            if not done:
                new_q = reward + self.discount * np.max(next_qs[i])
            # Once we hit game over state, there is no future_qs_list
            # So we just set new_q to reward
            else:
                new_q = reward

            X.append(current_state)
            y.append(new_q)

        self.model.fit(np.array(X), np.array(y), batch_size=self.minibatch_size, epochs=epochs, verbose=0)

        # Let exploration probability decay
        if self.epsilon > self.epsilon_min:
            # self.epsilon *= self.epsilon_decay
            self.epsilon -= self.epsilon_decay
