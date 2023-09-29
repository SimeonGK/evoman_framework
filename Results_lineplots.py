###############################################################################
# Evoman assignment 			                                              #
# Author: Julia Lammers        			                                      #
# 21-09-2023                    			                                  #
###############################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

runs = 10
save_dir = './results'

# Create a single figure with three subplots
fig, axs = plt.subplots(1, 3, figsize=(8, 2.75), constrained_layout=True, gridspec_kw={'wspace': 0.03, 'left': 0.04}, sharey=True, facecolor='whitesmoke')
fig.text(0.005, 0.5, 'Fitness', va='center', ha = 'center', rotation='vertical', fontsize=10)
fig.suptitle('Evolution of a population of specialist agents using 2 crossover strategies', fontsize=12, fontweight='bold', x=0.45, y=0.99, va='center')

# Loop over 3 enemies
for idx, enemy in enumerate([2, 5, 8]):
    # Store mean & best values per method in lists
    avg_mean_total = []
    std_mean_total = []
    avg_best_total = []
    std_best_total = []

    # Extract data per method out of result dictionaries and plot
    for method in (1,2):
        # Make a list of all directories with the results of 10 runs
        fitness_dir = []
        for run in range(1, runs+1):
            directory = f'method{method}_enemy{enemy}_run{run}'
            fitness_dir.append(directory)

        # Initialize empty dictionaries to store the data from each file
        dict = {directory: {"gen": [], "best": [], "mean": [], "std": []} for directory in fitness_dir}
        avg_mean = defaultdict(list)
        avg_best = defaultdict(list)
        avg_std = defaultdict(list)

        for directory in fitness_dir:
                file_path = os.path.join(directory,"results.txt")
                with open(file_path, "r") as file:
                    lines = file.readlines()

                    # Extract data from each line (if they contain floats)
                    for line in lines:
                        parts = line.strip().split()
                        try:
                            parts = [float(item) for item in parts]
                            gen, best, mean, std = map(float, parts)
                            dict[directory]["gen"].append(gen)
                            dict[directory]["best"].append(best)
                            dict[directory]["mean"].append(mean)
                            dict[directory]["std"].append(std)

                            avg_mean[gen].append(mean)
                            avg_best[gen].append(best)
                            avg_std[gen].append(std)

                        except ValueError:
                            pass

        # Calculate means per generation
        x_values = list(avg_mean.keys())
        avg_mean_values = [np.mean(avg_mean[gen]) for gen in x_values]
        std_mean_values = [np.std(avg_mean[gen]) for gen in x_values]
        avg_best_values = [np.mean(avg_best[gen]) for gen in x_values]
        std_best_values = [np.std(avg_best[gen]) for gen in x_values]

        # Store both methods in lists
        avg_mean_total.append(avg_mean_values)
        std_mean_total.append(std_mean_values)
        avg_best_total.append(avg_best_values)
        std_best_total.append(std_best_values)

    # Make plot per method
    width = 1
    height = 1.5

    # Create a line plot for method 1
    axs[idx].plot(x_values, avg_mean_total[0], label='Method 1 Mean', color='C4', linewidth=1)
    axs[idx].plot(x_values, avg_best_total[0], label='Method 1 Best', color='C6', linewidth=1)
    axs[idx].fill_between(x_values,
                          np.array(avg_mean_total[0]) - np.array(std_mean_total[0]),
                          np.array(avg_mean_total[0]) + np.array(std_mean_total[0]),
                          color='C4', alpha=0.2, linewidth=0.5)
    axs[idx].fill_between(x_values,
                          np.array(avg_best_total[0]) - np.array(std_best_total[0]),
                          np.array(avg_best_total[0]) + np.array(std_best_total[0]),
                          color='C6', alpha=0.2, linewidth=0.5)

    # Create a lineplot for method 2
    axs[idx].plot(x_values, avg_mean_total[1], label='Method 2 Mean', color='C0', linewidth=1)
    axs[idx].plot(x_values, avg_best_total[1], label='Method 2 Best', color='C2', linewidth=1)
    axs[idx].fill_between(x_values, np.array(avg_mean_total[1]) - np.array(std_mean_total[1]),
                          np.array(avg_mean_total[1]) + np.array(std_mean_total[1]),
                          color='C0', alpha=0.2, linewidth=0.5)
    axs[idx].fill_between(x_values, np.array(avg_best_total[1]) - np.array(std_best_total[1]),
                          np.array(avg_best_total[1]) + np.array(std_best_total[1]),
                          color='C2', alpha=0.2, linewidth=0.5)

    # Set plot labels and title
    axs[idx].set_xlim(0, 24)
    axs[idx].set_xlabel('Generation', fontsize=9, labelpad=3)
    axs[idx].set_ylim(0, 100)
    axs[idx].tick_params(axis='both', labelsize=7)
    axs[idx].set_title(f'Enemy {enemy}', fontsize=10)
    axs[idx].grid(True, linestyle='--', alpha=0.6)

    # Add legend in the third plot
    if idx == 2:
        axs[2].legend(fontsize=9, loc = 'lower right')

# Save and show the plot
fig.tight_layout()
fig.savefig(os.path.join(save_dir, 'Lineplot_comparison_EA_methods_over_enemies.png'), dpi=300, bbox_inches='tight')
fig.show()
