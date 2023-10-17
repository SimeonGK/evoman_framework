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
config = configparser.ConfigParser()

# Runs simulation
def simulation(env: Environment, genome) -> float:
    f, p, e, t = env.play(pcont=genome)
    return f


def main():
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    experiment_name = f'method_general_NEAT'  # Change the experiment name
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat.ini")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    save_path = os.path.join(local_dir, experiment_name)

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[1,2,3,4,5,6,7,8], #[1,2,3,7]
                    multiplemode="yes",
                    playermode="ai",
                    enemymode="static",
                    player_controller= NEAT_controller(config=config),
                    level=2,
                    speed="fastest",
                    visuals=False)
    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = simulation(env,genome)
    def run_neat(config):
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(1,filename_prefix=save_path))
        
        winner = p.run(eval_genomes, 10) #100 Generaties
    run_neat(config)






if __name__ == '__main__':
    main()