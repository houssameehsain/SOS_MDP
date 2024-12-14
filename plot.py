import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import glob
import os

# Set up plot style
sns.set()
sns.set_theme(style="darkgrid")
sns.set_context("paper")
sns.set_style("whitegrid")
plt.rcParams['grid.alpha'] = 0.3

def load_and_process_data(log_path):
    with open(log_path) as f:
        json_obj = json.load(f)
    
    returns = np.array(json_obj["returns"], dtype=np.float32)
    eps_lens = np.array(json_obj["eps_lengths"], dtype=np.float32)
    
    return returns, eps_lens

def smooth_data(data, window=100):
    """Smooth data using moving window average"""
    if len(data) < window:
        return data
    y = np.ones(window)
    z = np.ones(len(data))
    smoothed = np.convolve(data, y, 'same') / np.convolve(z, y, 'same')
    return smoothed

# Set figure size for horizontal layout (A4 width)
A4_WIDTH = 8.27
GOLDEN_RATIO = 1.618
fig_height = A4_WIDTH / GOLDEN_RATIO * 1.2  # Slightly taller to accommodate both rows

# Create figure with 2 rows (returns/lengths) and 3 columns (environments)
fig, axes = plt.subplots(2, 3, figsize=(A4_WIDTH, fig_height))
fig.suptitle('Training Performance Across Environments', fontsize=11, y=1.02)

# Define environments and algorithms
envs = ['gbrv0', 'gbrv1', 'gpnv0']
algos = ['dqn', 'ppo']
colors = {'dqn': '#1f77b4', 'ppo': '#ff7f0e'}
labels = {'dqn': 'DQN', 'ppo': 'PPO'}

# Track min episodes across all environments for consistent x-axis
min_episodes = float('inf')

# First pass to find minimum episode length
for env_idx, env in enumerate(envs):
    env_min_episodes = float('inf')
    for algo in algos:
        pattern = f"./new_logs/run_{env}_{algo}*/train/plots/performance_log.json"
        log_files = glob.glob(pattern)
        
        if log_files:
            for log_file in log_files:
                try:
                    returns, _ = load_and_process_data(log_file)
                    env_min_episodes = min(env_min_episodes, len(returns))
                except Exception as e:
                    print(f"Error processing {log_file}: {e}")
                    continue
    
    if env_min_episodes != float('inf'):
        min_episodes = min(min_episodes, env_min_episodes)

# Plot for each environment
for env_idx, env in enumerate(envs):
    col = env_idx  # Column index (0, 1, 2) for each environment
    
    for algo in algos:
        pattern = f"./new_logs/run_{env}_{algo}*/train/plots/performance_log.json"
        log_files = glob.glob(pattern)
        
        if not log_files:
            print(f"No log files found for {env} {algo}")
            continue
            
        all_returns = []
        all_eps_lens = []
        
        for log_file in log_files:
            try:
                returns, eps_lens = load_and_process_data(log_file)
                # Truncate to minimum episodes
                returns = returns[:min_episodes]
                eps_lens = eps_lens[:min_episodes]
                all_returns.append(returns)
                all_eps_lens.append(eps_lens)
            except Exception as e:
                print(f"Error processing {log_file}: {e}")
                continue
        
        if not all_returns:
            continue
            
        # Convert to numpy arrays
        all_returns = np.array(all_returns)
        all_eps_lens = np.array(all_eps_lens)
        
        # Calculate statistics
        mean_returns = np.mean(all_returns, axis=0)
        min_returns = np.min(all_returns, axis=0)
        max_returns = np.max(all_returns, axis=0)
        mean_eps_lens = np.mean(all_eps_lens, axis=0)
        min_eps_lens = np.min(all_eps_lens, axis=0)
        max_eps_lens = np.max(all_eps_lens, axis=0)
        
        # Apply smoothing
        smoothed_returns = smooth_data(mean_returns)
        smoothed_min_returns = smooth_data(min_returns)
        smoothed_max_returns = smooth_data(max_returns)
        smoothed_eps_lens = smooth_data(mean_eps_lens)
        smoothed_min_eps_lens = smooth_data(min_eps_lens)
        smoothed_max_eps_lens = smooth_data(max_eps_lens)
        
        # Create episode numbers
        episodes = np.arange(1, len(mean_returns) + 1)
        
        # Plot returns on top row
        axes[0, col].fill_between(episodes, 
                                smoothed_min_returns,
                                smoothed_max_returns,
                                color=colors[algo], alpha=0.1)
        axes[0, col].plot(episodes, smoothed_returns, 
                         color=colors[algo], label=labels[algo])
        
        # Plot episode lengths on bottom row
        axes[1, col].fill_between(episodes,
                                smoothed_min_eps_lens,
                                smoothed_max_eps_lens,
                                color=colors[algo], alpha=0.1)
        axes[1, col].plot(episodes, smoothed_eps_lens,
                         color=colors[algo], label=labels[algo])

# Update titles and labels
env_titles = ['Beady Ring v1', 'Beady Ring v2', 'Path Node v1']
for i, title in enumerate(env_titles):
    axes[0, i].set_title(title, fontsize=10, pad=8)
    axes[0, i].tick_params(labelsize=9)
    axes[1, i].tick_params(labelsize=9)
    axes[1, i].set_xlabel('Episode', fontsize=9)

# Add y-axis labels
for i in range(3):
    if i == 0:  # Only add y-labels to leftmost plots
        axes[0, i].set_ylabel('Average Return', fontsize=9)
        axes[1, i].set_ylabel('Average Episode Length', fontsize=9)

# Update legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 1.02),
          ncol=2, fontsize=9, frameon=True)

# Adjust spacing
plt.tight_layout()
plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.25)

# Save figure
plt.savefig("training_comparison.png", dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.close()

