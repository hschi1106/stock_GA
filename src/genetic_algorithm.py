import numpy as np
import itertools
from tqdm import tqdm

from matplotlib import pyplot as plt

class GeneticAlgorithm():
    def __init__(self, population_size, fitness_function, gene_cnt=0, gene_sample=[0, 1], selection_pressure=2, mutation_rate=0.01, patience=10, warmup=0, init_population=None):
        self.population_size = int(population_size)
        self.chromosomes = [] # list of chromosomes
        self.archive = [] # archive of some chromosomes

        self._selection_pressure = selection_pressure
        self._mutation_rate = mutation_rate

        self._gene_cnt = gene_cnt
        self._gene_types = gene_sample

        self._fitness_function = fitness_function

        self._patience = patience
        self._warmup = warmup

        if init_population is None:
            self.init_population()
        else:
            self.chromosomes = init_population
            self.chromosomes *= (self.population_size // len(init_population)) + 1
    
    def init_population(self):
        for i in range(self.population_size):
            self.chromosomes.append(np.random.choice(self._gene_types, size=(self._gene_cnt,)))

    def _fitness(self, chromosome): # fitness function
        return self._fitness_function(chromosome)
        
    def crossover(self):
        offspring = []
        np.random.shuffle(self.chromosomes)
        for i in range(0, len(self.chromosomes), 2):
            parent1 = self.chromosomes[i]
            
            if i + 1 >= len(self.chromosomes):
                offspring.append(parent1)
                break
            
            parent2 = self.chromosomes[i+1]
            crossover_point = np.random.randint(1, len(parent1))
            offspring.append(np.concatenate([parent1[:crossover_point], parent2[crossover_point:]]))
            offspring.append(np.concatenate([parent2[:crossover_point], parent1[crossover_point:]]))
            
        self.chromosomes = offspring
        
    def mutation(self):
        p_m = self._mutation_rate
        
        lengths = [len(chromosome) for chromosome in self.chromosomes]
        population_array = np.concatenate(self.chromosomes)
        
        pos = 0
        total_length = sum(population_array)
        
        while True:
            Pr = np.random.rand()
            
            t = int(np.floor(np.log(1 - Pr) / np.log(1 - p_m))) + 1
            
            pos += t
            if pos >= total_length:
                break
            
            population_array[pos] = np.random.choice(self._gene_types)
            
            pos += 1
            if pos >= total_length:
                break
        
        new_chromosomes = []
        start = 0
        for length in lengths:
            new_chromosomes.append(population_array[start:start+length])
            start += length
        
        self.chromosomes = new_chromosomes
    
    def tournament_selection(self):
        population_size = self.population_size
        selection_pressure = self._selection_pressure
        winners = []
        indices = np.arange(population_size)
        
        all_fitness = np.array([self._fitness(chromosome) for chromosome in self.chromosomes])
        
        if selection_pressure < 1:
            raise ValueError("Selection pressure must be >= 1")
        tournament_size = min(int(selection_pressure), population_size) if selection_pressure >= 2 else 2
        prob = selection_pressure - 1 if selection_pressure < 2 else None
        
        while len(winners) < population_size:
            participants = np.random.choice(indices, size=tournament_size, replace=False)
            if selection_pressure >= 2 or np.random.rand() < prob:
                winner_idx = participants[np.argmax(all_fitness[participants])]
            else:
                winner_idx = np.random.choice(participants)
            winners.append(self.chromosomes[winner_idx])

        self.chromosomes = winners

    def run(self, max_iter=100):
        avg_fitnesses, std_fitnesses = [], []
        avg_fitness_overall = -np.inf
        best_fitnesses = []
        no_improvement_count = 0

        for i in range(max_iter):
            # put archive chromosomes into the population
            # self.chromosomes.extend(self.archive)
            
            self.tournament_selection()
            self.crossover()
            self.mutation()

            fitnesses = [self._fitness(chromosome) for chromosome in self.chromosomes]

            avg_fitness = np.mean(fitnesses)
            avg_fitnesses.append(avg_fitness)
            std_fitness = np.std(fitnesses)
            std_fitnesses.append(std_fitness)

            best_fitness = np.argmax(fitnesses)
            best_fitnesses.append(fitnesses[best_fitness])
            
            # append the best 5% of chromosomes from this generation to the archive
            self.archive.extend(sorted(self.chromosomes, key=self._fitness, reverse=True)[:int(0.05 * self.population_size)])
            # keep the archive size to 5% of population size
            self.archive = sorted(self.archive, key=self._fitness, reverse=True)[:int(0.05 * self.population_size)]
                        
            # preset use science notation
            print(f"it: {i + 1}, es: {no_improvement_count}/{self._patience}, mean: {avg_fitness:+.9f}, std: {std_fitness:+.9f}, best: {fitnesses[best_fitness]:+.9f}, archive best: {self._fitness(self.archive[0]):+.9f}")
            if avg_fitness > avg_fitness_overall or avg_fitness < 0:
                avg_fitness_overall = avg_fitness
                no_improvement_count = 0
            else:
                if (i + 1) > self._warmup:
                    no_improvement_count += 1
                    if no_improvement_count >= self._patience:
                        print("Early stopping at iteration", i + 1)
                        break
        
        # self.chromosomes = self.chromosomes + self.archive
        
        # sort population by fitness
        fitnesses = [self._fitness(chromosome) for chromosome in self.chromosomes]
        sorted_idx = np.argsort(fitnesses, axis=0)[::-1]
        new_chromosomes = [self.chromosomes[i] for i in sorted_idx]
        new_chromosomes = [[int(gene) for gene in chromosome] for chromosome in new_chromosomes]
        self.chromosomes = new_chromosomes

        print("Max fitness: ", fitnesses[sorted_idx[0]])
        print("Chromosome: ", self.chromosomes[0])
        print("Archive best fitness: ", self._fitness(self.archive[0]))
        print("Archive best chromosome: ", self.archive[0])

        plt.plot(avg_fitnesses, color='blue', label='Average Fitness')
        plt.plot(best_fitnesses, color='red', label='Best Fitness')
        plt.fill_between(range(len(avg_fitnesses)), np.array(avg_fitnesses) - np.array(std_fitnesses), np.array(avg_fitnesses) + np.array(std_fitnesses), alpha=0.2)
        plt.fill_between(range(len(avg_fitnesses)), np.array(avg_fitnesses) - 2 * np.array(std_fitnesses), np.array(avg_fitnesses) + 2 * np.array(std_fitnesses), alpha=0.1)
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.title("Genetic Algorithm Fitness")
        plt.legend()
        plt.savefig("fitness.png")
        plt.clf()

        return self.chromosomes

class cGeneticAlgorithm(GeneticAlgorithm):
    def __init__(self, population_size, fitness_function, gene_cnt=0, gene_sample=[0, 1], selection_pressure=2, mutation_rate=0.01, patience=10, warmup=0, init_population=None, lr=0.1, eps=0.0001):
        super().__init__(population_size, fitness_function, gene_cnt, gene_sample, selection_pressure, mutation_rate, patience, warmup, init_population)
        self.prob_vector = np.array([[1/len(gene_sample)] * len(gene_sample) for _ in range(self._gene_cnt)])
        self.learning_rate = lr
        self.greedy_eplison = eps
        
        if init_population is None:
            print("No initial population provided, using random initialization.")
            # pass
        else:
            # sample from chromosomes
            self.chromosomes = init_population
            chroms = np.array(self.chromosomes)
            prob_vector = np.zeros((self._gene_cnt, len(self._gene_types)))
            for i in range(self._gene_cnt):
                # get the unique values and their counts
                unique, counts = np.unique(chroms[:, i], return_counts=True)
                # create a dictionary mapping unique values to their counts
                for j, val in enumerate(unique):
                    idcx = self._gene_types.index(val)
                    prob_vector[i][idcx] = counts[j] / len(self.chromosomes)
            self.prob_vector = prob_vector
            # print("Prob Vector: ", self.prob_vector)
    
    def compact_sample(self):
        # cal chromosome's inner probability

        # 1. cal prob vector
        chroms_np = np.array(self.chromosomes)
        prob_vector = np.zeros((self._gene_cnt, len(self._gene_types)))
        for i in range(self._gene_cnt):
            # get the unique values and their counts
            unique, counts = np.unique(chroms_np[:, i], return_counts=True)
            # create a dictionary mapping unique values to their counts
            for j, val in enumerate(unique):
                idcx = self._gene_types.index(val)
                prob_vector[i][idcx] = counts[j] / len(self.chromosomes)
        
        # 2. update prob vector
        self.prob_vector = (1 - self.learning_rate) * self.prob_vector + self.learning_rate * prob_vector
        self.prob_vector = (1 - self.greedy_eplison) * self.prob_vector + self.greedy_eplison * np.array([[1/len(self._gene_types)] * len(self._gene_types) for _ in range(self._gene_cnt)])
        
        # print("Prob Vector: ", self.prob_vector)

        # 3. sample new chromosome
        new_chromosomes = []
        for i in range(self.population_size):
            new_chromosome = []
            for j in range(self._gene_cnt):
                new_chromosome.append(np.random.choice(self._gene_types, p=self.prob_vector[j]))
            new_chromosomes.append(new_chromosome)
        
        # 4. update the chromosomes
        self.chromosomes = new_chromosomes

    def pv_entropy(self):
        # cal entropy of prob vector
        entropy = 0
        for i in range(self._gene_cnt):
            p = self.prob_vector[i]
            entropy -= np.sum(p * np.log2(p + 1e-10))
        return entropy

    def run(self, max_iter=100):
        avg_fitnesses, std_fitnesses = [], []
        avg_fitness_overall = -np.inf
        best_fitnesses = []
        no_improvement_count = 0

        for i in range(max_iter):
            self.tournament_selection()
            self.compact_sample()

            fitnesses = [self._fitness(chromosome) for chromosome in self.chromosomes]

            avg_fitness = np.mean(fitnesses)
            avg_fitnesses.append(avg_fitness)
            std_fitness = np.std(fitnesses)
            std_fitnesses.append(std_fitness)

            best_fitness = np.argmax(fitnesses)
            best_fitnesses.append(fitnesses[best_fitness])

            # preset use science notation
            print(f"it: {i + 1}, es: {no_improvement_count}/{self._patience}, mean: {avg_fitness:+.9f}, std: {std_fitness:+.9f}, best: {fitnesses[best_fitness]:+.9f}, entropy: {self.pv_entropy():+.9f}")
            if avg_fitness > avg_fitness_overall:
                avg_fitness_overall = avg_fitness
                no_improvement_count = 0
            else:
                if (i + 1) > self._warmup:
                    no_improvement_count += 1
                    if no_improvement_count >= self._patience:
                        print("Early stopping at iteration", i + 1)
                        break
        
        # sort population by fitness
        fitnesses = [self._fitness(chromosome) for chromosome in self.chromosomes]
        sorted_idx = np.argsort(fitnesses, axis=0)[::-1]
        new_chromosomes = [self.chromosomes[i] for i in sorted_idx]
        new_chromosomes = [[int(gene) for gene in chromosome] for chromosome in new_chromosomes]
        self.chromosomes = new_chromosomes

        print("Max fitness: ", fitnesses[sorted_idx[0]])
        print("Chromosome: ", self.chromosomes[0])

        plt.plot(avg_fitnesses, color='blue', label='Average Fitness')
        plt.plot(best_fitnesses, color='red', label='Best Fitness')
        plt.fill_between(range(len(avg_fitnesses)), np.array(avg_fitnesses) - np.array(std_fitnesses), np.array(avg_fitnesses) + np.array(std_fitnesses), alpha=0.2)
        plt.fill_between(range(len(avg_fitnesses)), np.array(avg_fitnesses) - 2 * np.array(std_fitnesses), np.array(avg_fitnesses) + 2 * np.array(std_fitnesses), alpha=0.1)
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.title("cGA Fitness")
        plt.legend()
        plt.savefig("fitness.png")

        return self.chromosomes

class ecGeneticAlgorithm(GeneticAlgorithm):
    def __init__(self, population_size, fitness_function, gene_cnt=0, gene_sample=[0, 1], selection_pressure=2, mutation_rate=0.01, patience=10, warmup=0, init_population=None):
        super().__init__(population_size, fitness_function, gene_cnt, gene_sample, selection_pressure, mutation_rate, patience, warmup, init_population)
        self.cache_dict = {}
        self.eplison = 0.01
    
    def extract_dict(self, bb_dict, partial_cnt=None):
        if partial_cnt is None:
            arr = np.array(self.chromosomes, dtype=int)
        else:
            partial_cnt = min(partial_cnt, len(self.chromosomes))
            sample_idx = np.random.choice(len(self.chromosomes), size=partial_cnt, replace=False)
            arr = np.array(self.chromosomes, dtype=int)[sample_idx]

        all_dict = {}
        for block in bb_dict:
            # test cache
            if tuple(block) in self.cache_dict:
                uniq, counts = self.cache_dict[tuple(block)]
            else:
                sub = arr[:, block]  # shape = (pop_n, len(block))
                uniq, counts = np.unique(sub, axis=0, return_counts=True)
                self.cache_dict[tuple(block)] = (uniq, counts)
            # 紀錄用於後面抽樣
            all_dict[tuple(block)] = {
                tuple(row): cnt for row, cnt in zip(uniq, counts)
            }
        return all_dict
        
    def cal_complexity(self, bb_dict, partial_cnt=None):
        if partial_cnt is None:
            pop_n = len(self.chromosomes)
            arr = np.array(self.chromosomes, dtype=int)
        else:
            # random sample partial_cnt chromosomes
            partial_cnt = min(partial_cnt, len(self.chromosomes))
            pop_n = partial_cnt
            sample_idx = np.random.choice(len(self.chromosomes), size=partial_cnt, replace=False)
            arr = np.array(self.chromosomes, dtype=int)[sample_idx]
        base = len(self._gene_types)
        log_factor = np.log2(pop_n)

        data_comp = 0
        all_dict = {}
        for block in bb_dict:
            # test cache
            if tuple(block) in self.cache_dict:
                uniq, counts = self.cache_dict[tuple(block)]
            else:
                sub = arr[:, block]  # shape = (pop_n, len(block))
                uniq, counts = np.unique(sub, axis=0, return_counts=True)
                self.cache_dict[tuple(block)] = (uniq, counts)
            # 紀錄用於後面抽樣
            all_dict[tuple(block)] = {
                tuple(row): cnt for row, cnt in zip(uniq, counts)
            }
            # 計算 entropy
            p = counts / pop_n
            entropy = -(p * np.log2(p)).sum()
            data_comp += entropy

        data_comp *= pop_n
        model_comp = sum(base**len(block) for block in bb_dict) * log_factor

        return model_comp, data_comp, all_dict
    
    def bb_operation(self):
        """
        Block‐building operation using MDL‐inspired merge:
        1. Start with each gene in its own block.
        2. Iteratively find the single best merge (i.e. the one that most reduces overall complexity),
           apply it, and repeat until no merge improves the score.
        3. Sample new chromosomes from the final block partition.
        Returns:
            best_pair_dict: dict mapping each block (as a tuple of indices) to its observed counts dict.
        """
        # 初始化
        n_genes = self._gene_cnt
        pop = len(self.chromosomes)
        best_blocks = [[i] for i in range(n_genes)]
        mc, dc, best_pair_dict = self.cal_complexity(best_blocks)
        best_score = mc + dc
        
        # 兩階段合併迴圈
        self.cache_dict = {}
        while True:
            curr = best_blocks
            curr_c = best_score

            sec_best_blocks = best_blocks
            sec_best_pair = best_pair_dict
            sec_best_score = best_score

            # 掃描 (i,j) 組合，找出最能降低分數的合併
            all_combinations = list(itertools.combinations(range(len(curr)), 2))
            comb_idx = np.random.choice(len(all_combinations), size=min(300000, len(all_combinations)), replace=False)
            sampled_combinations = [all_combinations[i] for i in comb_idx]

            for i, j in sampled_combinations:
                merged = sorted(curr[i] + curr[j])
                new_blocks = [blk for idx, blk in enumerate(curr) if idx not in (i, j)] + [merged]
                key = tuple(tuple(b) for b in sorted(new_blocks))
                mc, dc, pair_dict = self.cal_complexity(new_blocks, partial_cnt=None)
                score = mc + dc

                if score < sec_best_score:
                    sec_best_score = score
                    sec_best_blocks = new_blocks
                    sec_best_pair = pair_dict
                    break
            
            # 如果找到了改善，就一次性更新；否則跳出
            if sec_best_score < best_score:
                best_score = sec_best_score
                best_blocks = sec_best_blocks
                best_pair_dict = sec_best_pair
            else:
                break
        
        print("Best blocks:", best_blocks)
        # 抽樣新族群
        new_chromosomes = [[0] * n_genes for _ in range(pop)]
        for block in best_blocks:
            counts = best_pair_dict[tuple(block)]
            alleles = list(counts.keys()) + ['epsilon'] # add a random element
            freqs = np.array(list(counts.values()) + [0]) # add a random element
            probs = freqs / freqs.sum() # raw probabilities

            # add eplison greedy
            epsilon_probs = np.zeros(len(probs))
            epsilon_probs[-1] = 1
            probs = (1 - self.eplison) * probs + self.eplison * epsilon_probs

            # probs += 1e-3
            # probs /= probs.sum()
            picks = np.random.choice(len(alleles), size=pop, p=probs)
            samples = [alleles[idx] for idx in picks]

            for indi, gene_vals in enumerate(samples):
                if gene_vals == 'epsilon':
                    # random element
                    for pos in block:
                        new_chromosomes[indi][pos] = np.random.choice(self._gene_types)
                else:
                    for pos, val in zip(block, gene_vals):
                        new_chromosomes[indi][pos] = val

        self.chromosomes = new_chromosomes
        return best_pair_dict
    
    def run(self, max_iter=100):
        avg_fitnesses, std_fitnesses = [], []
        avg_fitness_overall = -np.inf
        best_fitnesses = []
        no_improvement_count = 0

        for i in range(max_iter):
            self.tournament_selection()
            bpd = self.bb_operation()

            fitnesses = [self._fitness(chromosome) for chromosome in self.chromosomes]

            avg_fitness = np.mean(fitnesses)
            avg_fitnesses.append(avg_fitness)
            std_fitness = np.std(fitnesses)
            std_fitnesses.append(std_fitness)

            best_fitness = np.argmax(fitnesses)
            best_fitnesses.append(fitnesses[best_fitness])

            # preset use science notation
            print(f"it: {i + 1}, es: {no_improvement_count}/{self._patience}, mean: {avg_fitness:+.9f}, std: {std_fitness:+.9f}, best: {fitnesses[best_fitness]:+.9f}")
            if avg_fitness > avg_fitness_overall or avg_fitness < 0:
                avg_fitness_overall = avg_fitness
                no_improvement_count = 0
            else:
                if (i + 1) > self._warmup:
                    no_improvement_count += 1
                    if no_improvement_count >= self._patience:
                        print("Early stopping at iteration", i + 1)
                        break
        
        # sort population by fitness
        fitnesses = [self._fitness(chromosome) for chromosome in self.chromosomes]
        sorted_idx = np.argsort(fitnesses, axis=0)[::-1]
        new_chromosomes = [self.chromosomes[i] for i in sorted_idx]
        new_chromosomes = [[int(gene) for gene in chromosome] for chromosome in new_chromosomes]
        self.chromosomes = new_chromosomes

        print("Max fitness: ", fitnesses[sorted_idx[0]])
        print("Chromosome: ", self.chromosomes[0])

        plt.plot(avg_fitnesses, color='blue', label='Average Fitness')
        plt.plot(best_fitnesses, color='red', label='Best Fitness')
        plt.fill_between(range(len(avg_fitnesses)), np.array(avg_fitnesses) - np.array(std_fitnesses), np.array(avg_fitnesses) + np.array(std_fitnesses), alpha=0.2)
        plt.savefig("fitness.png")

        return self.chromosomes