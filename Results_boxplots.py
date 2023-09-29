###############################################################################
# Evoman assignment 			                                              #
# Author: Julia Lammers        			                                      #
# 28-09-2023                    			                                  #
###############################################################################

import os
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

save_dir = './results'

# test with enemy 2
runs = 10

# Create a single figure with three subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for idx, enemy in enumerate([2, 5, 8]):
    allgains = []

    for i in (1, 2):
        gain = []
        directories = []

        for run in range(1, runs + 1):
            directory = f'method{i}_enemy{enemy}_run{run}'
            directories.append(directory)  # Use append to add directory1 to gain1

        for dir in directories:
            file_path = os.path.join(dir, "mean_gain.txt")

            with open(file_path, "r") as file:
                try:
                    float_value = float(file.read().strip())
                    gain.append(float_value)
                except ValueError:
                    print("Error: The file does not contain a valid float value.")

        allgains.append(gain)


    # Perform ANOVA test
    f_statistic, p_value = stats.f_oneway(*allgains)

    # Create boxplots on the respective subplot
    axs[idx].boxplot(allgains)
    axs[idx].set_xticks([1, 2])
    axs[idx].set_xticklabels(['Method 1', 'Method 2'])
    axs[idx].set_ylabel("Gain (Ep - Ee)")
    axs[idx].set_title(f'Enemy {enemy} boxplot comparison')

    # Print ANOVA results
    print(f'ANOVA for Enemy {enemy}:')
    print(f'F-statistic: {f_statistic}')
    print(f'p-value: {p_value}')

    # Perform post-hoc Tukey's HSD test if ANOVA is significant
    if p_value < 0.05:
        posthoc = pairwise_tukeyhsd(allgains)
        print(posthoc)
    else:
        print('not significant')

# Save the entire figure with subplots
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Boxplot_comparison_all_enemies.png'), dpi=300, bbox_inches='tight')
plt.show()