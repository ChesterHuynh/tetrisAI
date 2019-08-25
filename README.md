# TetrisAI
Implemented a Deep Q-learning Network to teach an agent how to play Tetris. The Q-learning algorithm is a reinforcement 
learning algorithm that uses the general concept of providing the agent rewards when it achieves what we want it to achieve 
and punishments otherwise. 

For the sake of keeping the animation short, here is the gameplay footage of roughly the first 50,000 points of the AI.

![Demo - First 10000 points](./demo.gif)

## Background
### Vanilla Q-learning Algorithm
Specifically, the Q-learning algorithm seeks to learn a policy that maximizes the total reward by interacting with the environment as well as constructing and updating a state-action table that lists all possible actions that can be taken in each possible state of the game. The table has the shape (num_states, num_actions) and each entry stores "quality values" (aka "Q-values"), which get updated as an action is committed by the agent.

For a given state that the agent is in, the agent chooses an action either by *exploring* a random action or *exploiting* information from our table and selecting the action with the highest q-value. We control how the agent decides on its next action by tweaking the probability of taking a random action `epsilon`, which is between 0 and 1. Often you want to let the agent explore random actions when starting from scratch, keeping `epsilon` close to or equal to 1 and gradually decay to `epsilon_min`, usually 0.01 or 0.

### Deep Q-Learning Networks


## Getting Started
### Interactive Version
I implemented an interactive version of tetris using Pygame to better understand the game mechanics in `tetris.py`.
You can run `$ python runinteractive.py` in command line to begin an interactive game.

The score for the interactive game is updated according to the following rule after each piece is placed: `score += (level // 2 + 1) * (1 + 10 * (rows_cleared ** 2)` and the game speed increases as the level increases. The player starts at level 0 and proceeds to subsequent levels after clearing 12 lines. Looking more closely at the update rule, the player gets 1 point times the level multiplier `level // 2 + 1`. When clearing lines, the player receives points equal to the board_width times the number of rows cleared squared times the level multiplier.


### AI Version
For the AI, a Deep Q-Learning Network (DQN) is implemented, aiming for a goal score of 5,000 points. The AI doesn't have levels, so the score is updated according to the following rule after each piece is placed: `score += 1 + 10 * (rows_cleared ** 2)`, similar to the interactive version's score update rule.


See how you stack up against the AI! You can run the AI version by running `$ python main.py` in command line to begin training the agent.


### Helpful Links:
- [Sentdex's Reinforcement YouTube Series](https://www.youtube.com/watch?v=yMk_XtIEzH8)
- [Article on the q-learning algorithm](https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56)
- [Article on the DQN framework](https://towardsdatascience.com/self-learning-ai-agents-part-ii-deep-q-learning-b5ac60c3f47)
- [More math on DQN's](https://towardsdatascience.com/dqn-part-1-vanilla-deep-q-networks-6eb4a00febfb)
- [DeepMind's Deep RL/DQN Paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [Paper on CS231n Tetris AI Project](https://docs.google.com/viewer?url=http%3A%2F%2Fcs231n.stanford.edu%2Freports%2F2016%2Fpdfs%2F121_Report.pdf)
