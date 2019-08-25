from tetris import Tetris
from DQN import DQNAgent, ModifiedTensorBoard
from datetime import datetime
from statistics import mean, median
from tqdm import tqdm
import time
import pdb
import os
import random

MIN_SCORE = 10000

def run_game():
    env = Tetris()
    episodes = 2000
    max_steps = None
    discount = 0.98
    replay_mem_size = 20000
    minibatch_size = 512
    epsilon = 1
    epsilon_min = 0
    epsilon_stop_episode = 1500
    learning_rate = 5e-3
    epochs = 1
    show_every = 50
    log_every = 50
    replay_start_size = 2000
    train_every = 1
    hidden_dims = [64, 64]
    activations = ['relu', 'relu', 'linear']

    agent = DQNAgent(env.get_state_size(), discount=discount, \
                   replay_mem_size=replay_mem_size, \
                   minibatch_size=minibatch_size, epsilon=epsilon, \
                   # epsilon_decay=epsilon_decay, \
                   epsilon_min=epsilon_min, \
                   epsilon_stop_episode=epsilon_stop_episode, \
                   learning_rate=learning_rate, hidden_dims=hidden_dims, \
                   activations=activations, \
                   replay_start_size=replay_start_size)

    log_dir = f'log/tetris-{datetime.now().strftime("%Y%m%d-%H%M%S")}-nn={str(hidden_dims)}-mem={replay_mem_size}-bs={minibatch_size}-discount={discount}'
    log = ModifiedTensorBoard(log_dir=log_dir)

    scores = []
    for episode in tqdm(range(episodes)):
        current_state = env.reset_game()
        done = False
        step = 0
        log.step = episode

        if show_every and episode % show_every == 0:
            show = True
        else:
            show = False

        # Run the game until either game over or we've hit max number of steps
        while not done and (not max_steps or step < max_steps):
            next_states = env.get_next_states()
            best_state = agent.best_state(next_states.values())

            best_action = None
            # action is (x,i), state is [lines_cleared, holes, total_bumpiness, sum_height]
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break

            # reward is the score, done is gameover status
            reward, done = env.play_game(best_action[0], best_action[1], show=show)
            if show:
                env.show()
            agent.update_replay_memory(current_state, best_action, next_states[best_action], reward, done)

            # move to next timestep
            current_state = next_states[best_action]
            step += 1
        if show:
            # After game is completed, collect the final score
            print("Episode %d  score: %d  epsilon: %.2f" % (episode, env.get_game_score(), agent.epsilon))
        scores.append(env.get_game_score())

        agent.train(epochs=epochs)

        if log_every and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])

            log.update_stats(avg_score=avg_score, min_score=min_score, max_score=max_score)

        if env.get_game_score() >= MIN_SCORE:
            if not os.path.exists('models/'):
                os.makedirs('models/')
            agent.model.save(f'models/eps_{str(episode)}nn_{str(hidden_dims)}__bs{minibatch_size}__score_{env.get_game_score()}__{int(time.time())}.h5')

if __name__ == "__main__":
    run_game()
