###############################################################################
# Evoman assignment 			                                              #
# Author: Julia Lammers        			                                      #
# 21-09-2023                    			                                  #
###############################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

plots_dir = './Plots_Task_II'
runs = 10
generations = 100
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

fig, axs = plt.subplots(1, 2, figsize=(8, 2.75), constrained_layout=True, gridspec_kw={'wspace': 0.03, 'left': 0.04}, sharey=True, facecolor='whitesmoke')
fig.text(0.005, 0.5, 'Fitness', va='center', ha = 'center', rotation='vertical', fontsize=10)
fig.suptitle("Population Evolution of Generalist Agents", fontsize=12, fontweight='bold',va='center')

directory_name = ['./Final_Results_Task_II','./Final_Results_Task_II_NEAT']
for idx, group in enumerate((('1 t/m 8'),('1,2,3,7'))):   # Now these are 2 results to compare, later we want these to be 2 groups of enemies
    avg_mean_total = []
    std_mean_total = []
    avg_best_total = []
    std_best_total = []
    groupnr = idx + 1

    #for method in (1,2): #This is going to be Neural network vs NEAT. Now it is only one method. # TODO
    for i in range(2):

        # Make a list of all directories with the results of 10 runs
        fitness_dir = []
        for run in range(1, runs+1):
            directory = f'{directory_name[i]}/TaskII_group{groupnr}_{run}'
            fitness_dir.append(directory)

        dict = {directory: {"gen": [], "best": [], "mean": [], "std": []} for directory in fitness_dir}
        avg_mean = defaultdict(list)
        avg_best = defaultdict(list)
        avg_std = defaultdict(list)

        for directory in fitness_dir:
            file_path = os.path.join(directory, "results.txt")
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

    axs[idx].plot(x_values, avg_mean_total[0], label='NN Mean', color='C4', linewidth=1)
    axs[idx].plot(x_values, avg_best_total[0], label='NN Best', color='C6', linewidth=1)
    axs[idx].fill_between(x_values,
                        np.array(avg_mean_total[0]) - np.array(std_mean_total[0]),
                        np.array(avg_mean_total[0]) + np.array(std_mean_total[0]),
                        color='C4', alpha=0.2, linewidth=0.5)
    axs[idx].fill_between(x_values,
                        np.array(avg_best_total[0]) - np.array(std_best_total[0]),
                        np.array(avg_best_total[0]) + np.array(std_best_total[0]),
                        color='C6', alpha=0.2, linewidth=0.5)
    # Create a lineplot for method 2
    axs[idx].plot(x_values, avg_mean_total[1], label='NEAT Mean', color='C0', linewidth=1)
    axs[idx].plot(x_values, avg_best_total[1], label='NEAT Best', color='C2', linewidth=1)
    axs[idx].fill_between(x_values, np.array(avg_mean_total[1]) - np.array(std_mean_total[1]),
                        np.array(avg_mean_total[1]) + np.array(std_mean_total[1]),
                        color='C0', alpha=0.2, linewidth=0.5)
    axs[idx].fill_between(x_values, np.array(avg_best_total[1]) - np.array(std_best_total[1]),
                        np.array(avg_best_total[1]) + np.array(std_best_total[1]),
                        color='C2', alpha=0.2, linewidth=0.5)
    # Set plot labels and title
    axs[idx].set_xlim(0, generations)
    axs[idx].set_xlabel('Generation ($\mathit{n}$)', fontsize=9, labelpad=3)
    axs[idx].set_ylim(0, generations)
    axs[idx].tick_params(axis='both', labelsize=7)
    axs[idx].set_title(f'Group {groupnr} ({group})', fontsize=10)
    axs[idx].grid(True, linestyle='--', alpha=0.6)

    axs[0].legend(fontsize=9, loc='upper left')

# Save and show the plot
#fig.tight_layout()
fig.savefig(os.path.join(plots_dir, 'Lineplot_TaskII.png'), dpi=300, bbox_inches='tight')
fig.show()

