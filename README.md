# TetrisAI
Standard Deep Q-learning Network approach to teach an agent how to play Tetris. The Q-learning algorithm is a reinforcement 
learning algorithm that uses the general concept of providing the agent rewards when it achieves what we want it to achieve 
and punishments otherwise.

### Interactive Version
I implemented an interactive version of tetris using Pygame to better understand the game mechanics in `tetris.py`.
You can run `$ python runinteractive.py` in command line to begin an interactive game.

The score for the interactive game is calculated as:
  `score += (level // 2 + 1) * (1 + rows_cleared ** 2)`
and the game speed is adjusted as the level increases.


### AI Version
For the AI, a Deep Q-Learning Network (DQN) is implemented, aiming for a goal score of 5,000 points. The AI doesn't have levels, so the score is calculated as:
  `score += (level // 2 + 1) * (1 + rows_cleared ** 2)`

See how you stack up against the AI! You can run the AI version by running `$ python main.py` in command line to begin training the agent.


### Helpful Links:
- [Sentdex's Reinforcement YouTube Series](https://www.youtube.com/watch?v=yMk_XtIEzH8)
- [Article on the q-learning algorithm](https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56)
- [Article on the DQN framework](https://towardsdatascience.com/self-learning-ai-agents-part-ii-deep-q-learning-b5ac60c3f47)
- [More math on DQN's](https://towardsdatascience.com/dqn-part-1-vanilla-deep-q-networks-6eb4a00febfb)
- [DeepMind's Deep RL Paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
