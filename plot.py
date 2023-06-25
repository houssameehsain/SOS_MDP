import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json


# data 1
f = open("./run_gpnv0_dqn_10_01/train/plots/performance_log.json")

# "./logs/run_gbrv0_dqn_moore/train/plots/gbrv0_dqn_moore_09_train_plots_performance_log.json"

json_obj = json.load(f)

returns_1 = np.array(json_obj["returns"], dtype=np.float32)[:22087]
eps_lens_1 = np.array(json_obj["eps_lengths"], dtype=np.float32)[:22087]
print(len(returns_1)) 

perf_1 = {
    'episode': [i + 1 for i in range(len(returns_1))],
    'return': returns_1,
    'eps_len': eps_lens_1,
}

indices_1 = np.array([i + 1 for i in range(len(returns_1))], dtype=np.int32)

perf_df_1 = pd.DataFrame(perf_1, index=indices_1)


# data 2
f = open("./run_gpnv0_ppo_10_01/train/plots/performance_log.json")
json_obj = json.load(f)

returns_2 = np.array(json_obj["returns"], dtype=np.float32)
eps_lens_2 = np.array(json_obj["eps_lengths"], dtype=np.float32)
print(len(returns_2))

perf_2 = {
    'episode': [i + 1 for i in range(len(returns_2))],
    'return': returns_2,
    'eps_len': eps_lens_2,
}

indices_2 = np.array([i + 1 for i in range(len(returns_2))], dtype=np.int32)

perf_df_2 = pd.DataFrame(perf_2, index=indices_2)


sns.set()
sns.set_theme(style="darkgrid")
sns.set_context("paper")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

# plot 1
smooth = 100
for metric in list(perf_1.keys()):
    if metric == 'episode':
        continue
    else:
        if metric == 'return':
            color = 'b'
            style = '-'
            op = 1
        else: 
            continue
        
        if smooth > 1:
            """
            smooth data with moving window average.
            that is,
                smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
            where the "smooth" param is width of that window (2k+1)
            """
            y = np.ones(smooth)
            x = np.asarray(perf_df_1[metric])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')

            # line plots
            ax1.plot(perf_df_1["episode"], x, c=color, ls=style, alpha=0.15)
            ax1.plot(perf_df_1["episode"], smoothed_x, c=color, ls=style, label="DQN", alpha=op)

        else:
            # line plot
            ax1.plot(perf_df_1["episode"], perf_df_1[metric], 'b-', label='value', alpha=1)

# plot 2
smooth = 100
for metric in list(perf_2.keys()):
    if metric == 'episode':
        continue
    else:
        if metric == 'return':
            color = 'orange'
            style = '-'
            op = 1
        else: 
            continue
        
        if smooth > 1:
            """
            smooth data with moving window average.
            that is,
                smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
            where the "smooth" param is width of that window (2k+1)
            """
            y = np.ones(smooth)
            x = np.asarray(perf_df_2[metric])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')

            # line plots
            ax1.plot(perf_df_2["episode"], x, c=color, ls=style, alpha=0.15)
            ax1.plot(perf_df_2["episode"], smoothed_x, c=color, ls=style, label="PPO", alpha=op)

            # if multi runs, can use this to display a 0.95 confidence interval
            # plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
        else:
            # line plot
            ax1.plot(perf_df_2["episode"], perf_df_2[metric], 'b-', label='value', alpha=1)

# plot 3
smooth = 100
for metric in list(perf_1.keys()):
    if metric == 'episode':
        continue
    else:
        if metric == 'eps_len':
            color = 'b'
            style = '-'
            op = 1
        else: 
            continue
        
        if smooth > 1:
            """
            smooth data with moving window average.
            that is,
                smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
            where the "smooth" param is width of that window (2k+1)
            """
            y = np.ones(smooth)
            x = np.asarray(perf_df_1[metric])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')

            # line plots
            ax2.plot(perf_df_1["episode"], x, c=color, ls=style, alpha=0.15)
            ax2.plot(perf_df_1["episode"], smoothed_x, c=color, ls=style, label="DQN", alpha=op)

        else:
            # line plot
            ax2.plot(perf_df_1["episode"], perf_df_1[metric], 'b-', label='value', alpha=1)

# plot 4
smooth = 100
for metric in list(perf_2.keys()):
    if metric == 'episode':
        continue
    else:
        if metric == 'eps_len':
            color = 'orange'
            style = '-'
            op = 1
        else: 
            continue
        
        if smooth > 1:
            """
            smooth data with moving window average.
            that is,
                smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
            where the "smooth" param is width of that window (2k+1)
            """
            y = np.ones(smooth)
            x = np.asarray(perf_df_2[metric])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')

            # line plots
            ax2.plot(perf_df_2["episode"], x, c=color, ls=style, alpha=0.15)
            ax2.plot(perf_df_2["episode"], smoothed_x, c=color, ls=style, label="PPO", alpha=op)

            # if multi runs, can use this to display a 0.95 confidence interval
            # plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
        else:
            # line plot
            ax2.plot(perf_df_2["episode"], perf_df_2[metric], 'b-', label='value', alpha=1)


handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.8, 1), loc='upper center', ncol=4)
# plt.legend()

ax1.set_xlabel('Episode')
ax1.set_ylabel('Average Cummulative Reward')

ax2.set_xlabel('Episode')
ax2.set_ylabel('Average Episode Length')

plt.suptitle('GPN Environment', fontsize=12, fontweight='bold')

# plt.show()
plt.savefig(fname="./gpnv0_dqn_ppo.png", dpi=300)

