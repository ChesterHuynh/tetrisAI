# TetrisAI
<p>Simple Deep Q-Learning Network approach to teach an agent how to play Tetris</p>


Implemented an interactive version of tetris using Pygame to better understand the game mechanics in `tetris.py`.
You can run `$ python runinteractive.py` in command line to begin an interactive game.

The score for the interactive game is calculated as:
  `score += (level // 2 + 1) * (1 + rows_cleared ** 2)`
and the game speed is adjusted as the level increases.

For the AI, a Deep Q-Learning Network (DQN) is implemented, aiming for a goal score of 5,000 points. The AI doesn't have levels, so the score is calculated as:
  `score += (level // 2 + 1) * (1 + rows_cleared ** 2)`

See how you stack up against the AI! You can run the AI version by running `$ python main.py` in command line to begin training the agent.



### Helpful Links:
-<a href="https://www.youtube.com/watch?v=yMk_XtIEzH8">Sentdex's Reinforcement YouTube Series</a>
