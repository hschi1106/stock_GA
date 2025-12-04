import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import coint
import os
import tqdm

paths = [f"./data/stock_tick/{d}" for d in os.listdir("./data/stock_tick") if os.path.isdir(f"./data/stock_tick/{d}")]
stock_ids = [os.path.basename(path) for path in paths]
stock_ids = stock_ids[:10]

all_pairs = []

# hh:mm:ss from 9:00:00 to 13:30:00 every 1s
time_range = pd.date_range(start="09:00:00", end="13:30:00", freq='10S').strftime('%H:%M:%S').tolist()
time_range = [time[:7] for time in time_range]  # keep only hh:mm
time_df = pd.DataFrame({'Time': time_range})

for f_stock_id in tqdm.tqdm(stock_ids, desc="Processing stock pairs"):
    for s_stock_id in stock_ids:
        if f_stock_id == s_stock_id:
            continue
        tick_f_dfs = []
        tick_s_dfs = []
        for date in pd.date_range(start="2025-06-18", end="2025-06-20"):
            date_str = date.strftime("%Y-%m-%d")
            try:
                tick_f_df = pd.read_csv(f"./data/stock_tick/{f_stock_id}/{date_str}.csv")
                tick_s_df = pd.read_csv(f"./data/stock_tick/{s_stock_id}/{date_str}.csv")
            except FileNotFoundError:
                print(f"No data for {f_stock_id} on {date_str}, skipping...")
                continue
            if tick_f_df.empty or tick_s_df.empty:
                print(f"Empty data for {f_stock_id} or {s_stock_id} on {date_str}, skipping...")
                continue
            tick_f_df['t'] = tick_f_df['Time'].str[:7]
            tick_f_df = tick_f_df.groupby(tick_f_df['t']).last().reset_index(drop=True)
            tick_s_df['t'] = tick_s_df['Time'].str[:7]
            tick_s_df = tick_s_df.groupby(tick_s_df['t']).last().reset_index(drop=True)
            
            # leave time front 7 characters
            tick_f_df['Time'] = tick_f_df['Time'].str[:7]
            tick_s_df['Time'] = tick_s_df['Time'].str[:7]
            
            # print(tick_f_df.head())
            
            # fill missing times by previous value
            
            tick_f_df = pd.merge(time_df.copy(), tick_f_df, on='Time', how='left').ffill().bfill()
            tick_s_df = pd.merge(time_df.copy(), tick_s_df, on='Time', how='left').ffill().bfill()
            
            tick_f_dfs.append(tick_f_df)
            tick_s_dfs.append(tick_s_df)
            
        df_f = pd.concat(tick_f_dfs)
        df_s = pd.concat(tick_s_dfs)
        
        len_f = len(df_f)
        len_s = len(df_s)
        if len_f / len_s < 0.2 or len_s / len_f < 0.2:
            continue
        # convert date and Time to datetime
        
        df_f['datetime'] = pd.to_datetime(df_f['date'] + ' ' + df_f['Time'])
        df_s['datetime'] = pd.to_datetime(df_s['date'] + ' ' + df_s['Time'])
        # sort by datetime
        df_f.sort_values('datetime', inplace=True)
        df_s.sort_values('datetime', inplace=True)
        # rename columns for clarity
        df_f.rename(columns={'deal_price': f'price_{f_stock_id}'}, inplace=True)
        df_s.rename(columns={'deal_price': f'price_{s_stock_id}'}, inplace=True)
        
        merge_df = pd.merge_asof(
            df_f[['datetime', f'price_{f_stock_id}']],
            df_s[['datetime', f'price_{s_stock_id}']],
            on='datetime',
            direction='backward'
        )
        
        merge_df.dropna(inplace=True)
        
        coint_t, p_value, crit_vals = coint(
            merge_df[f'price_{f_stock_id}'],
            merge_df[f'price_{s_stock_id}']
        )
        
        crit_90_val = crit_vals[2]
        if coint_t < crit_90_val:
            all_pairs.append([f_stock_id, s_stock_id, coint_t, p_value])
            
# sort all_pairs by p_value
all_pairs.sort(key=lambda x: x[3])
# save to csv
df_pairs = pd.DataFrame(all_pairs, columns=['stock_id_1', 'stock_id_2', 'coint_t', 'p_value'])
df_pairs.to_csv('cointegrated_pairs.csv', index=False) 
        