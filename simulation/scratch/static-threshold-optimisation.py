import sys
sys.dont_write_bytecode = True  # Prevent __pycache__ directory being created.

import os
import subprocess
import random
import statistics
import numpy as np
import time
import json

from utils import get_instantaneous_data, calculate_reward, get_total_pfc_pause

class GA:
    def __init__(self, pop_size, tournament_size, w):
        self.pop_size = pop_size
        self.tournament_size = tournament_size
        self.pm = 0.2
        self.pc = 0.8
        self.num_elites = 1
        self.num_generations = 10
        self.results_fp = f"../static_ecn_ga_w_{str(w)}.txt"

        self.population = None
        self.population_fitness = None
    
        self.PYTHON2_PATH = "/home/links/rl624/.conda/envs/hpcc_env/bin/python"

        
        self.lbounds = [0, 0, 0]
        self.ubounds = [2_000, 2_000, 1]  # KB
        self.std_devs = [0.05 * x for x in self.ubounds]  # std deviations to use to add gaussian noise for the mutation operation - 5%

        self.w = w
        self.alpha = 40

        self.trace_file = "60_incast_web_search_50_0.1s"
        self.topo = "60_incast"
        self.sim_args = {
            "cc": "dcqcn",
            "bw": 100,
            "topo": self.topo,
            "trace": self.trace_file,
            "w": str(w)
        }

        # N.B. the relevant results file is located at: mix/instant_qlen_{topo}_{trace}_{cc}{failure}_{rl_ecn_marking}.txt
        self.sim_results_file = f'mix/instant_qlen_{self.sim_args["topo"]}_{self.sim_args["trace"]}_{self.sim_args["cc"]}_0_{self.sim_args["w"]}.txt'
        self.run_history = []
        self.sim_cache = {}


    def start_ns3_simulation(self, k_min, k_max, p_max):
        if os.path.exists(self.sim_results_file):
            os.remove(self.sim_results_file)
        env = os.environ.copy()
        
        env["PATH"] = f"{os.path.dirname(self.PYTHON2_PATH)}:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"

        for key in ["CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_PYTHON_EXE", "PYTHON"]:
            env.pop(key, None)

        args = {
            **self.sim_args,
            "optimise_static_thresholds": 1,
            "k_min": k_min,
            "k_max": k_max,
            "p_max": p_max,
        }
        args_str = [f'--{key}={value}' for key, value in args.items()]
        args_str.append("--perform_eval")  # N.B: This only sets 'simulator_stop_time' in 'third.cc' script.

        subprocess.run([self.PYTHON2_PATH, "run.py", *args_str], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # This can be blocking as we need to wait for the whole simulation to run to get the average QLengths and throughput.

    def run(self):
        self.population = [[random.randint(1, 2500), random.randint(1, 2500), random.uniform(0, 1)]
                           for _ in range(self.pop_size)]
        run_stats = []
        print("STARTING GA RUN...", flush=True)
        for generation in range(self.num_generations):
            pop_fitness = []

            for i, chromosome in enumerate(self.population):
                k_min = chromosome[0]
                k_max = k_min + chromosome[1]
                p_max = chromosome[2]
                # print(f"{generation}:{i}", flush=True)
                if (k_min, k_max, p_max) in self.sim_cache:
                    fitness = self.sim_cache[(k_min, k_max, p_max)]
                    # print("HIT CACHE", flush=True)
                else:
                    # print("STARTING SIMULATION", flush=True)
                    self.start_ns3_simulation(k_min, k_max, p_max) # This is blocking - the fitness function simply reads the output file produced by the simulation.
                    fitness = self.get_fitness(chromosome)
                    self.sim_cache[(k_min, k_max, p_max)] = fitness
                    # print("FINISHED SIMULATION", flush=True)
                pop_fitness.append(fitness)
                print(f"GEN: {generation}, CHROMOSOME: {i}, Values: {chromosome}, fitness: {fitness}", flush=True)

            self.population_fitness = pop_fitness
            
            run_stats.append(self.get_fitness_stats())

            new_population_candidates = []
            while len(new_population_candidates) < (self.pop_size - self.num_elites):
                parents, fitnesses = self.selection()
                parent1, parent2 = parents
                parent1_fitness, parent2_fitness = fitnesses
                offspring = self.crossover(parent1, parent2, parent1_fitness, parent2_fitness)  
                mutated_offspring = self.mutation(offspring)
                new_population_candidates.append(mutated_offspring)
            
            pop_fitness_desc = sorted(enumerate(self.population_fitness), key=lambda x: x[1], reverse=True)
            elitest_chromosome_indices = [val[0] for val in pop_fitness_desc[:self.num_elites]]
            elitist_chromosome = [self.population[i] for i in elitest_chromosome_indices]
            self.population = new_population_candidates[:self.pop_size - self.num_elites] + elitist_chromosome
        best_candidate_idx = max(enumerate(self.population_fitness), key=lambda x: x[1])[0]

        self.run_history = run_stats

        return self.population[best_candidate_idx]
    
    def mutation(self, chromosome):
        mutated_chromosome = chromosome.copy()
        for i in range(len(chromosome)):
            if random.random() < self.pm:
                mutated_chromosome[i] += np.random.normal(0, self.std_devs[i])
                mutated_chromosome[i] = np.clip(mutated_chromosome[i], self.lbounds[i], self.ubounds[i])

                if i in [0,1]:
                    mutated_chromosome[i] = int(mutated_chromosome[i])
                
        return mutated_chromosome

    def get_fitness_stats(self):
        return {
            'best': max(self.population_fitness),
            'worst': min(self.population_fitness),
            'mean': statistics.mean(self.population_fitness),
            'stdev': statistics.stdev(self.population_fitness)
        }

    def selection(self):
        parents = []
        fitnesses = []
        for _ in range(2):  # Select two parents
            # Generate a list of random indices to use as the tournament
            tournament_indices = [random.randint(0, self.pop_size - 1) for _ in range(self.tournament_size)] 
            tournament = [self.population[i] for i in tournament_indices]
            tournament_fitness = [self.population_fitness[i] for i in tournament_indices]
            # find the index of the max value in tournament_fitness
            winner_idx = max(enumerate(tournament_fitness), key=lambda x: x[1])[0]
            tournament_winner = tournament[winner_idx]
            parents.append(tournament_winner)
            fitnesses.append(tournament_fitness[winner_idx])
        return parents, fitnesses

    def crossover(self, parent1, parent2, parent1_fitness, parent2_fitness):
        # ============== Uniform crossover - not good for genetic diversity here ============
        # if random.random() < self.pc:
        #     # Create a random binary mask (equivalent to flipping a coin)
        #     mask = [random.randint(0,1) for _ in range(len(parent1))] 
        #     # If mask value is 1, use the gene from parent1, else use the gene from parent2
        #     offspring = [parent1[i] if mask[i] == 1 else parent2[i] for i in range(len(mask))]
        #     if offspring[0] > offspring[1]:
        #         offspring[0], offspring[1] = offspring[1], offspring[0]
        #     return offspring
        # else:  # Return the 'better' parent so that the crossover operation always returns one solution.
        #     if parent1_fitness > parent2_fitness:
        #         return parent1
        #     else:
        #         return parent2

        if random.random() < self.pc:
            alpha = 0.5
            offspring = []
            for i in range(len(parent1)):
                lo, hi = sorted([parent1[i], parent2[i]])
                d = hi - lo
                val = random.uniform(lo - alpha * d, hi + alpha * d)
                val = np.clip(val, self.lbounds[i], self.ubounds[i])
                offspring.append(int(val) if i in [0, 1] else val)
            return offspring
        else:
            return parent1 if parent1_fitness >= parent2_fitness else parent2

    def get_fitness(self, chromosome):
        # PORT = 1
        # data = get_instantaneous_data(self.sim_results_file)
        # print(f'got data with length: {len(data)}', flush=True)
        # print(json.dumps(data[0], indent=4), flush=True)
        # acc_reward = sum([calculate_reward(list(x.values())[0][PORT], self.w, self.alpha)["reward"] for x in data])
        # print(f'acc_reward: {acc_reward}', flush=True)

        # Generate fct data by running script
        env = os.environ.copy()
        
        env["PATH"] = f"{os.path.dirname(self.PYTHON2_PATH)}:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"

        for key in ["CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_PYTHON_EXE", "PYTHON"]:
            env.pop(key, None)

        chromosome_fct_fp = f"../analysis/ga_fct/{chromosome[0]}_{chromosome[1]}_{chromosome[2]}.txt"
        with open(chromosome_fct_fp, 'w') as f:
            subprocess.run([self.PYTHON2_PATH, "../analysis/fct_analysis.py", "-p", f"fct_{self.topo}_{self.trace_file}_0"], env=env, stdout=f, stderr=f)
        
        results = []
        with open(chromosome_fct_fp, 'r') as f:
            lines = f.readlines()
            for line in lines:
                _, flow_size, fct_mid, fct_95, fct_99 = [float(x) for x in line.strip().split()]
                results.append({
                    "flow_size": flow_size,
                    "fct_mid": fct_mid,
                    "fct_95": fct_95,
                    "fct_99": fct_99
                })
        # Data is already sorted based on flow size.
        top_n = 3
        flow_w = 0.1
        mice_99 = sum([x["fct_99"] for x in results[:top_n]])/top_n
        elephant_mid = sum([x["fct_mid"] for x in results[-top_n:]]) / top_n
        
        fct_score = flow_w * mice_99 + (1 - flow_w) * elephant_mid

        pfc_fp = f"./mix/pfc_{self.topo}_{self.trace_file}_dcqcn.txt"
        total_pause_ns = get_total_pfc_pause(pfc_fp)
        unacceptable_total_pause = 10_000_000  # ns = 10_000 us
        pfc_multiplier = 1/unacceptable_total_pause

        fitness = -(fct_score * (1 + pfc_multiplier * total_pause_ns))

        print(f'fct_score: {fct_score}, mice_99: {mice_99}, elephant_mid: {elephant_mid}', flush=True)
        print(f'total_pause_ns: {total_pause_ns}', flush=True)

        return fitness # negative as a smaller value is better
    
    def write_stats(self):
        with open(self.results_fp, 'w') as f:
            f.write(f'generation,{",".join(list(self.run_history[0].keys()))}\n')
            for i, stat in enumerate(self.run_history):
                f.write(f'{i},{",".join([str(val) for val in stat.values()])}\n')


import multiprocessing
import sys

def run_ga(w):
    # with open(f'../static_ecn_optimisation_w_{str(w)}.txt', 'w') as log_file:
    #     sys.stdout = log_file
    #     sys.stderr = log_file
    try:
        ga = GA(pop_size=12, tournament_size=12, w=w)
        best_solution = ga.run()
        print(f"W: {w},BEST SOLUTION: {best_solution}", flush=True)
        ga.write_stats()
    except Exception as e:
        print(e, flush=True)


if __name__ == "__main__":
    processes = []
    # weights = [0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995]
    weights = [0.5]  # Should work about 1:1 ratio for throughput to queue length.
    for w in weights:
        run_ga(w)
    
    # for w in weights:
    #     p = multiprocessing.Process(target=run_ga, args=(w,), )
    #     p.start()
    #     processes.append(p)
    
    # for p in processes:
    #     p.join()


