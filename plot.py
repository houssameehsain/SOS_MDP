import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json



f = open("./logs/run_QRDQN_BRv0_2000_fullObs/train/plots/performance_log.json")
json_obj = json.load(f)

returns = np.array(json_obj["scores"], dtype=np.float32)
# eps_lens = np.array(json_obj["eps_lengths"], dtype=np.float32)
print(len(returns)) 

# returns = np.array(new_list, dtype=np.float32) 

perf = {
    'episode': [i + 1 for i in range(len(returns))],
    'return': returns,
    # 'eps_len': eps_lens,
}

indices = np.array([i + 1 for i in range(len(returns))], dtype=np.int32)

perf_df = pd.DataFrame(perf, index=indices)

sns.set()
smooth = 100

for metric in list(perf.keys()):
    if metric == 'episode':
        continue
    else:
        if metric == 'return':
            color = 'b'
            style = '-'
            op = 1
        # elif metric == 'return':
        #     color = 'g'
        #     style = '-'
        #     op = 1
        else: 
            continue
        # elif metric == 'adS':
        #     color = '#777b7e'
        #     style = '-'
        #     op = 0.5
        # elif metric == 'rcS':
        #     color = '#999da0'
        #     style = '-'
        #     op = 0.5
        # elif metric == 'pcS':
        #     color = '#787276'
        #     style = '-'
        #     op = 0.5
        # elif metric == 'raS':
        #     color = '#808588'
        #     style = '-'
        #     op = 0.5
        # elif metric == 'rdS':
        #     color = '#b9bbb6'
        #     style = '-'
        #     op = 0.5
        
        if smooth > 1:
            """
            smooth data with moving window average.
            that is,
                smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
            where the "smooth" param is width of that window (2k+1)
            """
            y = np.ones(smooth)
            x = np.asarray(perf_df[metric])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')

            # line plots
            plt.plot(perf_df["episode"], x, c=color, ls=style, alpha=0.3)
            plt.plot(perf_df["episode"], smoothed_x, c=color, ls=style, label="QR-DQN", alpha=op)

            # if multi runs, can use this to display a 0.95 confidence interval
            # plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
        else:
            # line plot
            plt.plot(perf_df["episode"], perf_df[metric], 'b-', label='value', alpha=1)


# f = open("./Python/logs/run_/train/plots/performance_log.json")
# json_obj = json.load(f)

# returns = np.array(json_obj["scores"], dtype=np.float32)
# # eps_lens = np.array(json_obj["eps_lengths"], dtype=np.float32)
# print(len(returns))

# # print(rdS.shape)
# # print(adS.shape)
# # print(planVals.shape)

# perf = {
#     'episode': [i + 1 for i in range(len(returns))],
#     'return': returns,
#     # 'eps_len': eps_lens,
# }

# indices = np.array([i + 1 for i in range(len(returns))], dtype=np.int32)

# perf_df = pd.DataFrame(perf, index=indices)

# smooth = 1000

# for metric in list(perf.keys()):
#     if metric == 'episode':
#         continue
#     else:
#         if metric == 'return':
#             color = 'b'
#             style = '-'
#             op = 1
#         # elif metric == 'return':
#         #     color = 'g'
#         #     style = '-'
#         #     op = 1
#         else: 
#             continue
#         # elif metric == 'adS':
#         #     color = '#777b7e'
#         #     style = '-'
#         #     op = 0.5
#         # elif metric == 'rcS':
#         #     color = '#999da0'
#         #     style = '-'
#         #     op = 0.5
#         # elif metric == 'pcS':
#         #     color = '#787276'
#         #     style = '-'
#         #     op = 0.5
#         # elif metric == 'raS':
#         #     color = '#808588'
#         #     style = '-'
#         #     op = 0.5
#         # elif metric == 'rdS':
#         #     color = '#b9bbb6'
#         #     style = '-'
#         #     op = 0.5
        
#         if smooth > 1:
#             """
#             smooth data with moving window average.
#             that is,
#                 smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
#             where the "smooth" param is width of that window (2k+1)
#             """
#             y = np.ones(smooth)
#             x = np.asarray(perf_df[metric])
#             z = np.ones(len(x))
#             smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')

#             # line plots
#             plt.plot(perf_df["episode"], x, c=color, ls=style, alpha=0.3)
#             plt.plot(perf_df["episode"], smoothed_x, c=color, ls=style, label="Distributed Reward", alpha=op)

#             # if multi runs, can use this to display a 0.95 confidence interval
#             # plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
#         else:
#             # line plot
#             plt.plot(perf_df["episode"], perf_df[metric], 'b-', label='value', alpha=1)



# f = open("./Python/logs/run_/train/plots/performance_log.json")
# json_obj = json.load(f)

# returns = np.array(json_obj["scores"], dtype=np.float32)
# # eps_lens = np.array(json_obj["eps_lengths"], dtype=np.float32)

# # print(rdS.shape)
# # print(adS.shape)
# # print(planVals.shape)

# print(len(returns)) 

# perf = {
#     'episode': [i + 1 for i in range(len(returns))],
#     'return': returns,
#     # 'eps_len': eps_lens,
# }

# indices = np.array([i + 1 for i in range(len(returns))], dtype=np.int32)

# perf_df = pd.DataFrame(perf, index=indices)

# smooth = 1000

# for metric in list(perf.keys()):
#     if metric == 'episode':
#         continue
#     else:
#         if metric == 'return':
#             color = 'r'
#             style = '-'
#             op = 1
#         # elif metric == 'return':
#         #     color = 'g'
#         #     style = '-'
#         #     op = 1
#         else: 
#             continue
#         # elif metric == 'adS':
#         #     color = '#777b7e'
#         #     style = '-'
#         #     op = 0.5
#         # elif metric == 'rcS':
#         #     color = '#999da0'
#         #     style = '-'
#         #     op = 0.5
#         # elif metric == 'pcS':
#         #     color = '#787276'
#         #     style = '-'
#         #     op = 0.5
#         # elif metric == 'raS':
#         #     color = '#808588'
#         #     style = '-'
#         #     op = 0.5
#         # elif metric == 'rdS':
#         #     color = '#b9bbb6'
#         #     style = '-'
#         #     op = 0.5
        
#         if smooth > 1:
#             """
#             smooth data with moving window average.
#             that is,
#                 smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
#             where the "smooth" param is width of that window (2k+1)
#             """
#             y = np.ones(smooth)
#             x = np.asarray(perf_df[metric])
#             z = np.ones(len(x))
#             smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')

#             # line plots
#             plt.plot(perf_df["episode"], x, c=color, ls=style, alpha=0.3)
#             plt.plot(perf_df["episode"], smoothed_x, c=color, ls=style, label="Non-Distributed Reward", alpha=op)

#             # if multi runs, can use this to display a 0.95 confidence interval
#             # plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
#         else:
#             # line plot
#             plt.plot(perf_df["episode"], perf_df[metric], 'b-', label='value', alpha=1)


plt.legend(title='legend')

plt.title('Generalized Beady Ring Environment', fontsize=14, fontweight='bold')
plt.xlabel('Episode')
plt.ylabel('Average Density Score')
plt.show()



