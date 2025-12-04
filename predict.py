import numpy as np
import pandas as pd
from tqdm import tqdm

import os, json, datetime

from src.indicator_lib import process_dataframe, indicator_filter
from src.utils import load_files, chromosome_extend

pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)

if __name__ == "__main__":
    # CONSTANTS
    chromosome_sets = ['short_0430', 'cond_short_48763', 'cond_short_16873']
    funds = 1000000 # 1 million
    margin_ratio = 1.3

    path_dayK_files, _ = load_files(left_range=1)
    stock_data_pds_all = []
    future_date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    for sfile in tqdm(path_dayK_files):
        df_dayK = pd.read_csv(sfile)
        if df_dayK.empty:
            continue
        df_dayK = df_dayK[['date', 'stock_id', 'open', 'close', 'spread', 'Trading_Volume', 'Trading_money', 'Trading_turnover', 'max', 'min']]
        df_dayK.loc[len(df_dayK)] = [future_date, df_dayK.iloc[-1]['stock_id'] , 1, 1, 1-df_dayK.iloc[-1]['close'], 0, 0, 0, 0, 0]
        # last 100 days for lower calculation (caution: this may affect ema)
        df_dayK = df_dayK.iloc[-200:]
        stock_data_pds_all.append(df_dayK)
    
    for chromosome_set in chromosome_sets:
        print("Strategy:", chromosome_set)
        cond_list = json.load(open(f"chromosomes/{chromosome_set}_cond.json", "r"))
        chromosomes = json.load(open(f"chromosomes/{chromosome_set}.json", "r"))
        
        idc_cnt = len(cond_list)
        stock_data_pds = []
        for sfile in stock_data_pds_all:
            df_dayK = process_dataframe(sfile, train=False, condition_list=cond_list)
            stock_data_pds.append(df_dayK)
        stock_data_pds = pd.concat(stock_data_pds, axis=0)

        # filter out the undecided indicators
        idc_cnt = 0
        for idx, (fp, sp, mul) in enumerate(cond_list):
            if fp == 'open' or sp == 'open' and chromosomes[0][idx] == 1:
                print(fp, ">", sp)
                chromosomes[0][idx] = 0
            if fp == 'open' or sp == 'open' and chromosomes[0][idx] == -1:
                print(fp, "<", sp)
                chromosomes[0][idx] = 0
        
        filtered_df = indicator_filter(stock_data_pds, chromosomes[0])
        filtered_df = filtered_df[['date', 'stock_id', 'sma5', 'sma20', 'sma60', 'upper', 'lower', 'prevc', 'vol', 'prevvol']]
        filtered_df = filtered_df[filtered_df['date'] == future_date]
        filtered_df = filtered_df[(filtered_df['prevc'] + 0.05) * 1000 * margin_ratio < funds] 
        filtered_df['funds_efficiency'] = (funds - (funds % ((filtered_df['prevc'] + 0.05) * 1000 * margin_ratio))) / funds
        filtered_df = filtered_df.sort_values(by='funds_efficiency', ascending=False)

        # filter prevc < 100 (0.1 M)
        # filtered_df = filtered_df[filtered_df['prevc'] < 100]
        # filter vol > 1000 000 (1000 lot)
        filtered_df = filtered_df[filtered_df['prevvol'] > 1000000]
        
        if filtered_df.empty:
            print("No prediction for today")
        else:
            print(filtered_df)
        


