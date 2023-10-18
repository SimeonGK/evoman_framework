###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys
from evoman.environment import Environment
from demo_controller import player_controller
from NEAT_controller import NEAT_controller
# imports other libs
import numpy as np
import os
import configparser
import neat
import pickle
import statistics
config = configparser.ConfigParser()

# Runs simulation
def simulation(env: Environment, genome) -> float:
    f, p, e, t = env.play(pcont=genome)
    return f, p, e



def main():
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat.ini")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    for nr, group in enumerate(([1,2,3,4,5,6,7,8],[1,2,3,7])):   # We compare two groups of enemies. 2nd group reflected a broad type of enemies.
        nr = nr+1
        for run_variable in range(1,11):     # 10 runs per agent
            if not os.path.exists('Final_Results_Task_II_NEAT_2'):
                os.makedirs('Final_Results_Task_II_NEAT_2')
            directory_name = f'TaskII_group{nr}_{run_variable}'
            experiment_name = f'./Final_Results_Task_II_NEAT_2/{directory_name}'
            if not os.path.exists(experiment_name):
                os.makedirs(experiment_name)

            env = Environment(experiment_name=experiment_name,
                            enemies=group, #[1,2,3,7]
                            multiplemode="yes",
                            playermode="ai",
                            enemymode="static",
                            player_controller= NEAT_controller(config=config),
                            level=2,
                            speed="fastest",
                            visuals=False)
            run_mode = 'test'
            #Function that evaluates all genomes
            def eval_genomes(genomes,config):
                for genome_id, genome in genomes:
                    genome.fitness = simulation(env,genome)[0]
            #Function that writes the appropiate statistics to the results.txt file
            def write_statistics_to_file(fitness_values, gen):
                file_path = os.path.join(experiment_name, 'results.txt')
                # Check if we need to write the header
                write_header = not os.path.exists(file_path)
                # Open the file for appending
                with open(file_path, 'a') as file:
                    # Write the header if the file didn't exist
                    if write_header:
                        file.write('gen best mean std\n')

                    best = max(fitness_values)
                    mean = statistics.mean(fitness_values)
                    std = statistics.stdev(fitness_values)

                    # Append the statistics to the file
                    file.write(f'{gen} {best:.6f} {mean:.6f} {std:.6f}\n')
                print(f"Statistics for gen {gen} written to {file_path}")
            #Function to run the NEAT algorithm
            def run_neat(config):
                stats = neat.StatisticsReporter()
                checkpointer = neat.Checkpointer(generation_interval=10, time_interval_seconds=None, filename_prefix=experiment_name+'/neat-checkpoint-')
                p = neat.Population(config)
                # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-3')
                p.add_reporter(checkpointer)
                p.add_reporter(stats)
                p.add_reporter(neat.StdOutReporter(True))
                
                winner = p.run(eval_genomes, 100) #100 Generaties
                #Save winner
                with open(experiment_name+"/best.pickle", "wb") as f:
                    pickle.dump(winner,f)
                # Save statistics
                stats.get_fitness_stat(write_statistics_to_file)
            
            if run_mode == 'test':
                        with open(experiment_name+"/best.pickle", 'rb') as f:
                            best_genome = pickle.load(f)
                        
                        print(f'\n RUNNING THE BEST SAVED SOLUTION FROM {experiment_name}')
                        # env.update_parameter('speed', 'normal')
                        gain = []

                        # 5 test-rounds of the best result
                        for i in range(5):
                            f,p,e = simulation(env,best_genome)
                            gain.append(p-e)

                        mean_gain = np.mean(gain)

                        gain_file = open(experiment_name+'/mean_gain.txt','w')
                        gain_file.write(str(mean_gain))
                        gain_file.close()
            # run_neat(config)
    






if __name__ == '__main__':
    main()