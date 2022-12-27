import os
import numpy as np


def log_plot_data(rewards, returns, eps_lengths, run):

    path = f"plot_data/{run.id}/"
    if not os.path.exists(path):
        os.makedirs(path)

    # log rewards
    arr = np.asarray(rewards)
    file = path + "rewards.npy"
    np.save(file, arr)

    # log returns
    arr = np.asarray(returns)
    file = path + "returns.npy"
    np.save(file, arr)

    # log episode lengths
    arr = np.asarray(eps_lengths)
    file = path + "eps_lengths.npy"
    np.save(file, arr)

    print(f"plot data saved to {path}")

def log_img():
    raise NotImplementedError
