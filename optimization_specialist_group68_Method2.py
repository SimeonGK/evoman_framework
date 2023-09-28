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

# imports other libs
import numpy as np
import os


# Runs simulation
def simulation(env: Environment, x: np.ndarray) -> tuple[float, int, int]:
    f, p, e, t = env.play(pcont=x)
    return f, p, e

# Evaluation
def evaluate(env: Environment, x: np.ndarray) -> np.ndarray:
    result_tuples = list(map(lambda y: simulation(env, y), x))
    f_values, p_values, e_values = zip(*result_tuples)

    f_array = np.array(f_values)
    p_array = np.array(p_values)
    e_array = np.array(e_values)

    return f_array, p_array, e_array


# Normalizes a value within the range [0, 1]
def normalize(population_fitness: np.ndarray, x: float) -> float:
    max_val = max(population_fitness)
    min_val = min(population_fitness)
    
    if max_val == min_val:
        raise ValueError("max_val should not be equal to min_val")
    
    normalized_x = (x - min_val) / (max_val - min_val)
    return max(0, min(1, normalized_x))

# Blend Crossover is a way of creating offspring in a region that is
# bigger than the (n-dimensional) rectangle spanned by the parents.
def blend_crossover(pop: np.ndarray, alpha: int) -> np.ndarray:
    offspring = np.zeros((0, pop.shape[1]))

    for i in range(0, pop.shape[0], 2):
        parent1 = pop[np.random.randint(0, 99)]
        parent2 = pop[np.random.randint(0, 99)]
        child1 = np.empty_like(parent1)
        child2 = np.empty_like(parent2)

        # Pick every value of child in range [p1 - (α * d), p2 + (α * d)]
        for j in range(pop.shape[1]):
            # Determine who is lower and upper parent
            # print("ch\n")
            x = min(parent1[j], parent2[j])
            y = max(parent1[j], parent2[j])
            distance = y - x
            # print("eck\n")

            # Calculate range
            l_limmit = x - (alpha * distance)
            u_limmit = y + (alpha * distance)
            # Select value in range
            # print("x=", x,"y=", y,"distance=", distance,"l=", l_limmit,"u=", u_limmit)
            child1[j] = np.random.uniform(l_limmit, u_limmit)
            child2[j] = np.random.uniform(l_limmit, u_limmit)

        offspring = np.vstack((offspring, child1, child2))

    return offspring

# Mutation opration: non-uniform mutation with Gaussian distribution on the offspring. Mutation probability = 100%, with
#   mutation step size (sigma) 0.1.
def mutation(offspring: np.ndarray, mutation_rate: float = 1, sigma: float = 0.1) -> np.ndarray:
    mutated_offspring = offspring.copy()  # Create a copy to avoid modifying the original array
    mask = np.random.random(offspring.shape) < mutation_rate  # Generate a random mask for mutation
    noise = np.random.normal(0, sigma, offspring.shape)  # Generate random noise from a Gaussian distribution
    mutated_offspring[mask] += noise[mask]  # Apply mutation to selected elements
    return mutated_offspring


# Tournament selection. Can be deleted, right?
def tournament(pop: np.ndarray) -> np.ndarray:
    gladiator1 = pop[np.random.randint(0, pop.shape[0])]
    gladiator2 = pop[np.random.randint(0, pop.shape[0])]

# Selection operator: Round Robin Tournament
def round_robin(pop: np.ndarray, population_fitness: np.ndarray, q: int) -> np.ndarray:
    round_robin = np.zeros(pop.shape[0])
    for i in range(len(pop)):
        contenters = np.random.choice(pop.shape[0], size=q, replace=False) #draw q contenders
        score = np.sum(population_fitness[contenters] < population_fitness[i])
        round_robin[i] = score
        indices_of_highest_scores = np.argsort(round_robin)[::-1][:100] #get the top 100
        # print(indices_of_highest_scores)
    return indices_of_highest_scores

    
def main():
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    for loop_variable in (2,5,8):   # We compare 3 enemies
        for run_variable in range(1,11):     # 10 runs per agent
            experiment_name = f'method2_enemy{loop_variable}_run{run_variable}'  # Change the experiment name

            if not os.path.exists(experiment_name):
                os.makedirs(experiment_name)

            n_hidden_neurons = 10

            # initializes simulation in individual evolution mode, for single static enemy.
            env = Environment(experiment_name=experiment_name,
                            enemies=[loop_variable],
                            playermode="ai",
                            player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                            enemymode="static",
                            level=2,
                            speed="fastest",
                            visuals=False)


            # number of weights for multilayer with 10 hidden neurons
            n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
            #Global variables
            upper_weight = 1
            lower_weight = -1
            npop = 100
            gens = 25
            run_mode = 'test'

            # Function to test an experiment.
            if run_mode == 'test':
                best_solution = np.loadtxt(experiment_name + "/best.txt")
                print(f'\n RUNNING THE BEST SAVED SOLUTION FROM {experiment_name}')
                env.update_parameter('speed', 'normal')
                gain = []

                # 5 test-rounds of the best result
                for i in range(5):
                    f,p,e = evaluate(env,[best_solution])
                    gain.append(p[0]-e[0])

                mean_gain = np.mean(gain)

                gain_file = open(experiment_name+'/mean_gain.txt','w')
                gain_file.write(str(mean_gain))
                gain_file.close()

                #sys.exit(0)


            # Evolution
            if not os.path.exists(experiment_name+'/evoman_solstate'):

                print( '\nNEW EVOLUTION\n')

                pop = np.random.uniform(lower_weight,upper_weight, (npop, n_vars))
                population_fitness,p,e = evaluate(env,pop)  # P and E are the energy points of the player (P) and enemy (E)
                best = np.argmax(population_fitness)
                mean = np.mean(population_fitness)
                std = np.std(population_fitness)
                gen_number = 0
                solutions = [pop,population_fitness]
                env.update_solutions(solutions)
            else:

                print('\n CONTINUING EVOLUTION FROM PREVIOUS RUN')

                env.load_state()
                pop = env.solutions[0]
                population_fitness = env.solutions[1]

                best = np.argmax(population_fitness)
                mean = np.mean(population_fitness)
                std = np.std(population_fitness)

                gen_file = open(experiment_name+'/gen.txt', 'r')
                gen_number = int(gen_file.readline()) # the generation number is stored on first line of gen.txt
                gen_file.close()

            # save the results (evaluation) from the first random initialisation population
            results_file = open(experiment_name+"/results.txt",'a')
            results_file.write('\n\ngen best mean std')
            print(f'\n GENERATION {str(gen_number)} {str(round(population_fitness[best],6))} {str(round(mean,6))} {str(round(std,6))}')
            results_file.write(f'\n{str(gen_number)} {str(round(population_fitness[best],6))} {str(round(mean,6))} {str(round(std,6))}')
            results_file.close()

            # Evolution loop for gens times
            for i in range(gen_number+1,gens):

                offspring = blend_crossover(pop, 0.5)
                mutated_offspring = mutation(offspring,1,0.1)
                offspring_fitness, p, e = evaluate(env,mutated_offspring)
                # create 1 population of offspring and population so steady state
                pop = np.vstack((pop,mutated_offspring))
                population_fitness = np.append(population_fitness,offspring_fitness)

                # Selection mechanism
                best_100 = round_robin(pop,population_fitness, 10)
                pop = pop[best_100]
                population_fitness = population_fitness[best_100]
                # Getting stats
                best = np.argmax(population_fitness)
                mean = np.mean(population_fitness)
                std = np.std(population_fitness)

                # Saving results
                results_file = open(experiment_name+"/results.txt",'a')
                print(f'\n GENERATION {str(i)} {str(round(population_fitness[best],6))} {str(round(mean,6))} {str(round(std,6))}')
                results_file.write(f'\n{str(i)} {str(round(population_fitness[best],6))} {str(round(mean,6))} {str(round(std,6))}')
                results_file.close()


                # Saving gen number
                gen_file = open(experiment_name+'/gen.txt','w')
                gen_file.write(str(i))
                gen_file.close()

                # Saving best solution
                np.savetxt(experiment_name+'/best.txt',pop[best])

                # Saving state
                solutions = [pop, population_fitness]
                env.update_solutions(solutions)
                env.save_state()


if __name__ == '__main__':
    main()