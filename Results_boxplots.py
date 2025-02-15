###############################################################################
# Evoman assignment 			                                              #
# Author: Julia Lammers        			                                      #
# 27-09-2023                    			                                  #
###############################################################################

import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

save_dir = './results'
runs = 10

# Create a single figure with three subplots
fig, axs = plt.subplots(1, 3, figsize=(8, 2.75), constrained_layout=True, gridspec_kw={'wspace': 0.03, 'left': 0.04}, sharey=True, facecolor='whitesmoke')
fig.text(0.001, 0.5, 'Gain ($\mathit{Ep - Ee}$)', va='center', ha='center', rotation='vertical', fontsize=10)
fig.suptitle('Comparison of Specialist Agent Performance Using 2 Evolutionary Algorithms', fontsize=12, fontweight='bold', x=0.45, y=0.99, va='center')

# Loop over 3 enemies
for idx, enemy in enumerate([2, 5, 8]):
    all_gains = []

    for i in (1, 2):
        gains = []
        directories = []

        for run in range(1, runs + 1):
            directory = f'method{i}_enemy{enemy}_run{run}'
            directories.append(directory)

        for dir in directories:
            file_path = os.path.join(dir, "mean_gain.txt") # mean_gain.txt file contains 1 float, which is the mean result over 5 test runs

            with open(file_path, "r") as file:
                try:
                    float_value = float(file.read().strip())
                    gains.append(float_value)
                except ValueError:
                    print("Error: The file does not contain a valid float value.")

        all_gains.append(gains)

    # Perform ANOVA test
    f_statistic, p_value = stats.f_oneway(*all_gains)

    # Print ANOVA results
    print(f'ANOVA for Enemy {enemy}:')
    print(f'F-statistic: {f_statistic}')
    print(f'p-value: {p_value}')

    # Perform post-hoc Tukey's HSD test if ANOVA is significant
    if p_value < 0.05:
        posthoc = pairwise_tukeyhsd(np.concatenate(all_gains), np.repeat(['Method 1', 'Method 2'], runs))
        print(posthoc)
    else:
        print('not significant')

    # Create boxplots on the respective subplot
    bp = axs[idx].boxplot(all_gains, patch_artist=True)

    # Set colors for boxplot 1 and boxplot 2
    for i, box in enumerate(bp['boxes']):
        if i == 0:
            box.set(facecolor='C6', alpha=0.7)
        elif i == 1:
            box.set(facecolor='C2', alpha=0.7)

    for median in bp['medians']:
        median.set(color='black', linewidth=1)

    # Add plot labels an titles
    axs[idx].set_xticks([1, 2])
    axs[idx].set_xticklabels(['Method 1', 'Method 2'])
    axs[idx].tick_params(axis='y', labelsize=7)
    axs[idx].set_title(f'Enemy {enemy}', fontsize=10)
    axs[idx].grid(True, linestyle='--', alpha=0.6)

# Save and show the plot
fig.tight_layout()
fig.savefig(os.path.join(save_dir, 'Boxplot_comparison_all_enemies.png'), dpi=300, bbox_inches='tight')
fig.show()
