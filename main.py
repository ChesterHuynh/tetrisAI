from tetris import Tetris
from DQN import DQNAgent, ModifiedTensorBoard
from datetime import datetime
from statistics import mean, median
from tqdm import tqdm
import random
import pdb


def run_game():
    env = Tetris()
    episodes = 5000
    max_steps = None
    replay_mem_size = 20000
    minibatch_size = 64
    epsilon = 0.95
    epsilon_min = 1e-3
    epsilon_decay = 0.99
    learning_rate = 1e-3
    epochs = 1
    show_every = 50
    log_every = 50
    replay_start_size = 2000
    train_every = 1
    hidden_dims = [64, 32]
    activations = ['relu', 'relu', 'linear']

    agent = DQNAgent(env.get_state_size(), replay_mem_size=replay_mem_size, \
                   minibatch_size=minibatch_size, epsilon=epsilon, \
                   epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, \
                   learning_rate=learning_rate, hidden_dims=[64,32], \
                   activations=activations, replay_start_size=replay_start_size)

    log_dir = f'log/tetris-{datetime.now().strftime("%Y%m%d-%H%M%S")}-nn={str(hidden_dims)}-mem={replay_mem_size}-bs={minibatch_size}-epochs={epochs}'
    log = ModifiedTensorBoard(log_dir=log_dir)

    scores = []
    for episode in tqdm(range(episodes)):
        current_state = env.reset_game()
        done = False
        steps = 0

        if show_every and episode % show_every == 0:
            show = True
        else:
            show = False

        while not done and (not max_steps or steps < max_steps):
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
            agent.update_replay_memory(current_state, next_states[best_action], reward, done)

            # move to next timestep
            current_state = next_states[best_action]
            steps += 1
        if show:
            print()
            print(env.board)
        # After game is completed
        scores.append(env.get_game_score())

        if episode % train_every == 0:
            agent.train(minibatch_size=minibatch_size, epochs=epochs)

        if log_every and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])

            log.update_stats(avg_score=avg_score, min_score=min_score, max_score=max_score)

if __name__ == "__main__":
    run_game()
