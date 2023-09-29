###############################################################################
# Evoman assignment 			                                              #
# Author: Julia Lammers        			                                      #
# 21-09-2023                    			                                  #
###############################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# Directories containing results.txt per method
### We must use 10 results.txt files per method

save_dir = './results'

for enemy in (2,5,8):
    #method1 = []
    #method2 = []

    avg_mean_total = []
    std_mean_total = []
    avg_best_total = []
    std_best_total = []

    for method in (1,2):
        fitness_dir = []
        for run in range(1, 11):
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
        std_best_values = [np.mean(avg_std[gen]) for gen in x_values]

        avg_mean_total.append(avg_mean_values)
        std_mean_total.append(std_mean_values)
        avg_best_total.append(avg_best_values)
        std_best_total.append(std_best_values)

    width = 3.25
    height = 4.5
    plt.figure(figsize=(width, height), facecolor='#F0F0F0')
    # Create a line plot for method 1
    plt.plot(x_values, avg_mean_total[0], label='Method 1 Mean', color='C4', linewidth=1)
    plt.plot(x_values, avg_best_total[0], label='Method 1 Best', color='C6', linewidth=1)
    plt.fill_between(x_values, np.array(avg_mean_total[0]) - np.array(std_mean_total[0]), np.array(avg_mean_total[0]) + np.array(std_mean_total[0]),
                     color='C4', alpha=0.2,linewidth=0.5)
    plt.fill_between(x_values, np.array(avg_best_total[0]) - np.array(std_best_total[0]), np.array(avg_best_total[0]) + np.array(std_best_total[0]),
                     color='C6', alpha=0.2, linewidth=0.5)
    #plt.errorbar(x_values, avg_mean_total[0], yerr=std_mean_values, fmt='none', ecolor='black',
                 #label='Standard Deviation')
    #plt.errorbar(x_values, avg_best_total[0], yerr=std_best_values, fmt='none', ecolor='black', label='Standard Deviation')


    #Create a lineplot for method 2
    plt.plot(x_values, avg_mean_total[1], label='Method 2 Mean', color='C0', linewidth=1)
    plt.plot(x_values, avg_best_total[1], label='Method 2 Best', color='C2', linewidth=1)
    plt.fill_between(x_values, np.array(avg_mean_total[1]) - np.array(std_mean_total[1]), np.array(avg_mean_total[1]) + np.array(std_mean_total[1]),
                     color='C0', alpha=0.2, linewidth=0.5)
    plt.fill_between(x_values, np.array(avg_best_total[1]) - np.array(std_best_total[1]), np.array(avg_best_total[1]) + np.array(std_best_total[1]),
                     color='C2', alpha=0.2, linewidth=0.5)

    # Set plot labels and title
    plt.xlim(0, 25)
    plt.xlabel('Generation', fontsize=9, labelpad=3)
    plt.ylim(0, 120)
    plt.ylabel('Fitness', fontsize=9, labelpad=1)
    plt.tick_params(axis='both', labelsize=7)
    plt.title(f'Enemy {enemy} EA method comparison', fontsize=10, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Add a legend to distinguish different lines
    plt.legend(fontsize=9)
    plt.savefig(os.path.join(save_dir, f'Lineplot method comparison enemy {enemy}.png'), dpi=300, bbox_inches='tight')

"""
for dataset_name, dataset in data_dict.items():
    plt.plot(dataset['gen'], dataset['mean'], label=dataset_name)


# Set plot labels and title
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Fitness over time')

# Add a legend to distinguish different datasets
plt.legend()

# Show the plot
plt.show()



#### TO DO:

- Het gemiddelde van alle lijnen plotten ipv alle lijnen los.
- STD toevoegen (we hebben maar 1 std, kan ik die gebruiken voor zowel best als mean?) Std over generaties ?
- Ook best plotten.
- Vragen: wat als niet alle runs evenveel generaties hebben?

- Overwegen: Dictionary anders indelen als bovenstaande niet lukt. 
"""