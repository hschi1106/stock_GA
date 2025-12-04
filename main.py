import numpy as np
import pandas as pd
from tqdm import tqdm

import os, json, datetime

from src.genetic_algorithm import GeneticAlgorithm, ecGeneticAlgorithm, cGeneticAlgorithm
from src.indicator_lib import process_dataframe, indicator_filter, gen_condition_list, compute_upi_jit
from src.utils import load_files, chromosomes_extend

from matplotlib import pyplot as plt

if __name__ == "__main__":
    # CONSTANTS
    chromosome_set = 'new_espx'
    direction = False # True for long, False for short
    discount_rate = 0.3 # 70% discount on transaction fee
    cond_list = "" #'top80_condlist'
    cond_list_seed = 123
    file_load_seed = 678

    # condition list
    if (cond_list != ""):
        assert os.path.exists(f"./cond_list/{cond_list}.json"), f"Condition list {cond_list} not found."
        cond_list = json.load(open(f"./cond_list/{cond_list}.json", "r"))
    else:
        cond_list = gen_condition_list(seed=cond_list_seed)
        
    idc_cnt = len(cond_list)
    print("Conditions:\n", cond_list, "\nCondition count:", idc_cnt)

    # file preprocessing
    path_dayK_files, _ = load_files(left_range=0.3, seed=file_load_seed)
    path_inst_files = "./data/stock_institution/"
    stock_data_pds = []
    for sfile in tqdm(path_dayK_files):
        df_dayK = pd.read_csv(sfile)

        # merge institution data
        df_inst = pd.read_csv(os.path.join(path_inst_files, os.path.basename(sfile)))
        if df_dayK.empty or df_inst.empty:
            print(f"Empty data found in {os.path.basename(sfile)}")
            continue
        f_missing_dates = df_dayK[~df_dayK['date'].isin(df_inst['date'])]['date']
        s_missing_dates = df_inst[~df_inst['date'].isin(df_dayK['date'])]['date']
        if not f_missing_dates.empty or not s_missing_dates.empty:
            print(f"Missing data found in {os.path.basename(sfile)}: {len(f_missing_dates)} in dayK, {len(s_missing_dates)} in inst")
            continue
        df_inst = df_inst.drop(columns=['stock_id'])
        df_dayK = df_dayK.merge(df_inst, on='date', how='left')
        # end of merge

        df_dayK = process_dataframe(df_dayK, train=True, condition_list=cond_list)
        stock_data_pds.append(df_dayK)
    
    stock_data_pds = pd.concat(stock_data_pds, axis=0)
    
    # fitness function
    def fitness(chromosome):
        new_data = indicator_filter(stock_data_pds, chromosome)
        
        # new_data = new_data[new_data['prevc'] < 100]
        new_data = new_data[new_data['vol'] > 1000000]
        
        rule_cnt = np.sum(np.array(chromosome) == 1) + np.sum(np.array(chromosome) == -1)
        rule_reward = -0.0001 * rule_cnt
        if new_data.empty:
            return rule_reward
        
        # calculate daily mean gain
        new_data['gain'] = np.clip(new_data['gain'], -0.1, 0.1)
        daily_mean = new_data.groupby('date')['gain'].mean()
        daily_mean = daily_mean if direction else -daily_mean
        penalty = 1 - np.exp(daily_mean.size / 800)

        # calculate geometric mean
        geomatric_mean = np.exp(np.mean(np.log1p(daily_mean))) - 1

        # random noise
        noise = 0.0003 * np.random.randn()

        return geomatric_mean * penalty + rule_reward + noise
    
    # init population
    # initp = [np.random.choice([-1, 0, 1], size=(idc_cnt,)) for _ in range(500)]
    # initp = json.load(open(f"chromosomes/{chromosome_set}.json", "r"))
    # print("Init Population:", len(initp))

    ga = cGeneticAlgorithm(
        population_size=500,
        fitness_function=fitness,
        gene_cnt=idc_cnt,
        gene_sample=[-1, 0, 1],
        selection_pressure=1.5,
        mutation_rate=1e-3,
        patience=30,
        warmup=200,
        init_population=None,
        lr=0.2,
        eps=1e-4,
    )
    
    chroms = ga.run(max_iter=5000)

    json.dump(chroms, open(f"chromosomes/{chromosome_set}.json", "w"))
    json.dump(cond_list, open(f"chromosomes/{chromosome_set}_cond.json", "w"))
