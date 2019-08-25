# TetrisAI
Implemented a Deep Q-learning Network to teach an agent how to play Tetris. The Q-learning algorithm is a reinforcement 
learning algorithm that uses the general concept of providing the agent rewards when it achieves what we want it to achieve 
and punishments otherwise. 

For the sake of keeping the animation short, here is the gameplay footage of roughly the first 50,000 points of the AI.

![Demo - First 10000 points](./demo.gif)

This project was done solely in **Python**. I created an interactive version of the game that utilizes the *PyGame* package as well as the AI version that utilizes the **Keras** framework and **TensorFlow** backend. 


## Getting Started and Implementation
### Interactive Version
I implemented an interactive version of tetris using Pygame to better understand the game mechanics in `tetris.py`.
You can run `$ python runinteractive.py` in command line to begin an interactive game.

The score for the interactive game is updated according to the following rule after each piece is placed: `score += (level // 2 + 1) * (1 + 10 * (rows_cleared ** 2)` and the game speed increases as the level increases. The player starts at level 0 and proceeds to subsequent levels after clearing 12 lines. Looking more closely at the update rule, the player gets 1 point times the level multiplier `level // 2 + 1`. When clearing lines, the player receives points equal to `board_width` times `rows_cleared` squared times the level multiplier.

The data structure of choice to represent the pieces were rectangular 2D lists, which has zeros for elements in the list that should be colored the same as the background and nonzero values that follow the shape of the piece. To keep majority of the code algorithmic, I only store one orientation of each block and algorithmically applied rotations and joins of pieces. As a result, in the display, you might see a delay in the rendering of certain colored squares when rectangular arrays overlap and the values in the arrays must be updated. 

### AI Version
For the AI, a Deep Q-Learning Network (DQN) is implemented, aiming for a goal score of 5,000 points. The AI doesn't have levels, so the score is updated according to the following rule after each piece is placed: `score += 1 + 10 * (rows_cleared ** 2)`, similar to the interactive version's score update rule. The data structure used to represent the pieces follow similarly to the interactive version as well.

The ANN used to construct the DQN is a `Sequential` model with two hidden layers, each with 32 neurons. The activations for each of the hidden layers are `ReLU's` and the activation for the last layer is `Linear`. The loss function used was `mean squared error` for our optimizer `Adam`.

The hyperparameters that had to be tuned for the DQN were as follows:
- `discount`: how strongly to consider future rewards over immediate ones; 0.98 is used.
- `replay_mem_size`: number of ***<state, new_state, action, reward>*** tuples stored in the ANN's "memory"; 20000 is used.
- `minibatch_size`: how large the random sample from the `replay_memory` the ANN should use to train over; 512 is used.
- `epsilon_stop_episode`: epsiode number when we reduce the exploration probability `epsilon` to `epsilon_min`; `epsilon` is set to 1, `epsilon_min` is set to 0, and `epsilon_stop_episode` is set to 75% of `episodes`.
- `learning_rate`: learning rate for our optimizer `Adam`; 0.005 is used.
- `hidden_dims`: number of hidden layers and the number of neurons in each layer; [64, 64] is used.

See how you stack up against the AI! You can run the AI version by running `$ python main.py` in command line to begin training the agent.


## Background
### Vanilla Q-learning Algorithm
Specifically, the Q-learning algorithm seeks to learn a policy that maximizes the total reward by interacting with the environment as well as constructing and updating a state-action table that lists all possible actions that can be taken in each possible state of the game. The table has the shape (num_states, num_actions) and each entry stores "quality values" (aka "Q-values"), which get updated as an action is committed by the agent.

For a given state that the agent is in, the agent chooses an action either by ***exploring*** a random action or ***exploiting*** information from our table and selecting the action with the highest q-value. We control how the agent decides on its next action by tweaking the probability of taking a random action `epsilon`, which is between 0 and 1. Often you want to let the agent explore random actions when starting from scratch, keeping `epsilon` close to or equal to 1 and gradually decay to `epsilon_min`, usually 0.01 or 0.

Updates on the q-values in the state-action table occur after each action and ends when an episode is completed (in this case, when a game over occurs). After each action is taken by the agent, the agent "interacts" with the environment, and the environment returns a reward in response to the action. The update rule is as follows:

`Q[state, action] = Q[state,action] + lr * (reward + discount * np.max(Q[new_state,:]) - Q[state, action]),` 

where
- `lr` is the learning rate, controls how much the q-value is affected by the results of the most recent action, i.e. how much you accept the new value vs. the old value. 
- `discount` is the discount factor, which is a value between 0 and 1 that controls how strongly the agent considers future values over immediate values. From our update rule, we see that the future reward is multiplied our variable `discount`. Typically, `discount` is between 0.8 and 0.99.
- `reward` is the value returned the environment in response to the agent taking an action.


### Deep Q-Learning Networks (DQN)
For environments with very large or infinitely large number of possible states and/or actions, using a table can be computationally expensive and usually impractical. Instead, we approximate the q-values that would be in a state-action table with an artificial neural network (ANN) or convolutional neural network (CNN). 

In this project, an ANN was used over a CNN since CNN's are better suited when the input data that must have its spatial/temporal ordering preserved, e.g. images or audio clips. To guarantee stability during training and preventing correlation between steps in a given episode, we construct two ANN's, one that is the Q-network and one that is the `target` network. 

To get q-values to update properly in Q-network, we need something to serve as the "validation data" for our algorithm to optimize over. The `target` network serves this purpose. When the Q-network is training and updating its weights to output q-values for each possible action given various features about the state of the environment, after every `update_target_every` number of episodes, the weights of the Q-network are copied and frozen into the `target` network for the next `update_target_every` number of episodes and is used to generate our "validation data". For DQN's, we generally optimize this by minimizing *mean squared error* and update the target model every 5-10 games. 

In addition, the Q-network also has a ***experience replay***, which in this project is a deque of size `memory_size` and has entries of the form ***<s, s', a, r>***, where s is the current state, s' is the new state, a is the action taken to get from s to s' and r is the reward from the environment in response to the action taken. While training, to provide greater stability in how the agent learns, the agent randomly samples from its experience replay, using old and new experiences alike. 


### Helpful Links:
- [Sentdex's Reinforcement YouTube Series](https://www.youtube.com/watch?v=yMk_XtIEzH8)
- [Article on the q-learning algorithm](https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56)
- [Article on the DQN framework](https://towardsdatascience.com/self-learning-ai-agents-part-ii-deep-q-learning-b5ac60c3f47)
- [More math on DQN's](https://towardsdatascience.com/dqn-part-1-vanilla-deep-q-networks-6eb4a00febfb)
- [DeepMind's Deep RL/DQN Paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [Paper on CS231n Tetris AI Project](http://cs231n.stanford.edu/reports/2016/pdfs/121_Report.pdf)
