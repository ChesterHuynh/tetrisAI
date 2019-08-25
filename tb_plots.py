from numpy import genfromtxt
import matplotlib.pyplot as plt
import pdb

fig, axs = plt.subplots(1, 3, figsize=(20, 10))

path_to_avg = './tensorboard_csvs/run-tetris-20190824-103449-nn=[64, 64]-mem=20000-bs=512-discount=0.98-tag-avg_score.csv'
path_to_max = './tensorboard_csvs/run-tetris-20190824-103449-nn=[64, 64]-mem=20000-bs=512-discount=0.98-tag-max_score.csv'
path_to_min = './tensorboard_csvs/run-tetris-20190824-103449-nn=[64, 64]-mem=20000-bs=512-discount=0.98-tag-min_score.csv'

avg_data = genfromtxt(path_to_avg, delimiter=',')
max_data = genfromtxt(path_to_max, delimiter=',')
min_data = genfromtxt(path_to_min, delimiter=',')

axs[0].plot(avg_data[1:,1], avg_data[1:,2])
axs[0].set_title("Average Scores Every 50 Episodes")
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Score")

axs[1].plot(max_data[1:,1], max_data[1:,2])
axs[1].set_title("Max Scores Every 50 Episodes")
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Score")

axs[2].plot(min_data[1:,1], min_data[1:,2])
axs[2].set_title("Min Scores Every 50 Episodes")
axs[2].set_xlabel("Episode")
axs[2].set_ylabel("Score")

plt.savefig("tensorboard_plots.svg")
