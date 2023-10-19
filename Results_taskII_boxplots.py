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

save_dir = './Plots_Task_II'
directory_name = ['./Final_Results_Task_II','./Final_Results_Task_II_NEAT']
runs = 10

# Create a single figure with three subplots
fig, axs = plt.subplots(1, 2, figsize=(4, 2.75), constrained_layout=True, gridspec_kw={'wspace': 0.03, 'left': 0.04}, sharey=True, facecolor='whitesmoke')
fig.text(0.001, 0.5, 'Gain ($\mathit{\u03A3(Ep - Ee)}$)', va='center', ha='center', rotation='vertical', fontsize=10)
fig.suptitle('Comparison of Generalist Agent Performance', fontsize=12, fontweight='bold', va='center')

# Store results per method
methods = [[],[]]

# Loop over 3 enemies
for idx, group in enumerate((('1 t/m 8'),('1,2,3,7'))):
    groupnr = idx + 1

    # Store results per group
    all_gains = []

    for i in range(2): # Method
        gains = []
        directories = []

        for run in range(1, runs + 1):
            directory = f'{directory_name[i]}/TaskII_group{groupnr}_{run}'
            directories.append(directory)

        for dir in directories:
            file_path = os.path.join(dir, "mean_gain.txt") # mean_gain.txt file contains 1 float, which is the mean result over 5 test runs
            with open(file_path, "r") as file:
                try:
                    float_value = float(file.read().strip())
                    gains.append(float_value)
                    print(gains)
                except ValueError:
                    print("Error: The file does not contain a valid float value.")

        # Append results
        all_gains.append(gains)
        methods[i].append(gains)

    ### STATISTICAL TESTING ###
    # Perform ANOVA test to compare methods per group
    f_statistic, p_value = stats.f_oneway(*all_gains)

    # Print ANOVA results
    print(f'ANOVA for Group {group}:')
    print(f'F-statistic: {f_statistic}')
    print(f'p-value: {p_value}')

    # Perform post-hoc Tukey's HSD test if ANOVA is significant
    if p_value < 0.05:
        posthoc = pairwise_tukeyhsd(np.concatenate(all_gains), np.repeat(['Method 1', 'Method 2'], runs))
        print(posthoc)
    else:
        print('not significant')

    ### BOXPLOTS ###
    # Create boxplots on the respective subplot
    bp = axs[idx].boxplot(all_gains, patch_artist=True, widths=0.4)

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
    axs[idx].set_xticklabels(['NN', 'NEAT'])
    axs[idx].tick_params(axis='y', labelsize=7)
    axs[idx].set_title(f'Group {group}', fontsize=10)
    axs[idx].grid(True, linestyle='--', alpha=0.6)


### STATISTICAL TESTING #2 ###
# Perform ANOVA test to compare groups per method  ## methods[0] = NN, Methods[1] = NEAT
f_statistic, p_value = stats.f_oneway(*methods[1])

# Print ANOVA results
print(f'ANOVA for NEAT')
print(f'F-statistic: {f_statistic}')
print(f'p-value: {p_value}')

# Perform post-hoc Tukey's HSD test if ANOVA is significant
if p_value < 0.05:
    posthoc = pairwise_tukeyhsd(np.concatenate(methods[1]), np.repeat(['1t/m8', '1,2,3,7'], runs))
    print(posthoc)
else:
    print('not significant')

# Save and show the plot
fig.savefig(os.path.join(save_dir, 'Boxplot_TaskII.png'), dpi=300, bbox_inches='tight')
fig.show()
