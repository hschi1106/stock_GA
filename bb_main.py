import numpy as np
import pandas as pd
from tqdm import tqdm

import os, json, datetime

from src.genetic_algorithm import ecGeneticAlgorithm, GeneticAlgorithm, cGeneticAlgorithm

if __name__ == "__main__":
    idc_cnt = 1000
    gene_sample = [-1, 0, 1]

    num_of_bits = 30
    traps_cnt = 18
    traps_k = 3

    # random sample 3 x 20 from num_of_BBs
    # trap_sample = np.random.choice(num_of_bits, size=(traps_cnt * traps_k, ), replace=True)
    trap_sample = np.array(list(range(num_of_bits)))
    # random shuffle
    np.random.shuffle(trap_sample)
    print("Trap Sample:", trap_sample)
    input("Press Enter to continue...")
    
    # fitness function
    def fitness(chromosome):
        chromosome = np.array(chromosome)
        identical_to_ones = (chromosome[trap_sample] == 0)
        identical_to_ones = (~identical_to_ones).reshape(-1, traps_k).astype(int).sum(axis=1)
        identical_to_ones[identical_to_ones == 0] = traps_k + 4

        return identical_to_ones.sum()

    # init population
    initp = [np.random.choice(gene_sample, size=(num_of_bits,)) for _ in range(1000)]

    ga = cGeneticAlgorithm(
        population_size=1000,
        fitness_function=fitness,
        gene_cnt=num_of_bits,
        gene_sample=gene_sample,
        selection_pressure=8, # 2000^(1/10) ~=2.1
        mutation_rate=8e-3, # 1/125 = 0.008
        patience=50,
        warmup=200,
        init_population=initp,
    )
    
    chroms = ga.run(max_iter=10000)
    # print("Best Prob Dict:", bpd)
