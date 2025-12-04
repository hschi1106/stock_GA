import numpy as np
import pandas as pd
from tqdm import tqdm

import os, json

from src.indicator_lib import process_dataframe, indicator_filter
from src.utils import load_files

from matplotlib import pyplot as plt

if __name__ == "__main__":
    # CONSTANTS
    leverage = 1
    direction = False # True for long, False for short
    discount_rate = 0.3 # 70% discount on transaction fee
    chromosome_set = 'new_espx'

    cond_list = json.load(open(f"chromosomes/{chromosome_set}_cond.json", "r"))
    idc_cnt = len(cond_list)

    _, path_dayK_files = load_files(left_range=0.3)
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

        df_dayK = process_dataframe(df_dayK, train=False, condition_list=cond_list)
        stock_data_pds.append(df_dayK)
    stock_data_pds = pd.concat(stock_data_pds, axis=0)

    best_chromosome = json.load(open(f"chromosomes/{chromosome_set}.json", "r"))

    filtered_dfx = indicator_filter(stock_data_pds, best_chromosome[0])

    results = []
    resulult_pick_winrate = []
    resultst_day_winrate = []
    resultst_month_backset = []
    for sl in np.arange(0, 0.1, 0.001):
        stop_loss = sl
        filtered_df = filtered_dfx.copy()
        # tax fee 0.003 (0.3%) in multiple day 0.0015 (0.15%) for one day
        # transaction fee 0.001425 (0.1425%) for BUY and SELL so (0.1425% * 2 = 0.285%)
        # discount 0.3 (30%) for Discount on the transaction fee
        # leverage 1.11x for short, 2.5x for long in TAIEX
        if direction:
            max_down = (filtered_df['min'] - filtered_df['open']) / filtered_df['open']
            filtered_df['gain'] = np.where(max_down < -stop_loss, -stop_loss, filtered_df['gain'])
            filtered_df['gain'] = ((filtered_df['gain']) - (1.425 * 2 * discount_rate + 1.5)/1000) * leverage
        else:
            max_up = (filtered_df['max'] - filtered_df['open']) / filtered_df['open']
            filtered_df['gain'] = np.where(max_up > stop_loss, stop_loss, filtered_df['gain'])
            filtered_df['gain'] = ((-filtered_df['gain']) - (1.425 * 2 * discount_rate + 1.5)/1000) * leverage

        # daily statistics
        day_trys, day_wins = 0, 0
        day_gains = []

        # monthly statistics
        month_tags = []
        month_cumprod_gains = []

        # group by month (YYYY-MM)
        filtered_df['date_m'] = filtered_df['date'].apply(lambda x: x[:7])
        grouped_month_df = filtered_df.groupby('date_m')
        for month_date, month_df in grouped_month_df:
            # for every month
            month_tags.append(month_date)

            # group by date
            grouped_day_df = month_df.groupby('date')
            one_month_gains = []
            for day_date, day_df in grouped_day_df:
                day_mean = day_df['gain'].mean()
                one_month_gains.append(day_mean)
                day_gains.append(day_mean)
                day_trys += 1
                day_wins += 1 if day_mean > 0 else 0
            month_cumprod_gain = (1 + np.array(one_month_gains)).cumprod()[-1]
            month_cumprod_gains.append(month_cumprod_gain - 1)

        pick_wins = sum([1 for gain in filtered_df['gain'] if gain > 0])
        pick_trys = len(filtered_df['gain'])
        
        array_gain = [100]
        for gain in month_cumprod_gains:
            array_gain.append(array_gain[-1] * (1 + gain))
        results.append(array_gain[-1] - 100)
        resulult_pick_winrate.append((pick_wins / pick_trys) * 100)
        resultst_day_winrate.append((day_wins / day_trys) * 100)
        resultst_month_backset.append(min(month_cumprod_gains) * 100)

    # 2x2 subplot
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(np.arange(0, 0.1, 0.001), results, label='Cumprod Gain')
    ax[0, 0].set_title('Cumprod Gain')
    ax[0, 0].set_xlabel('Stop Loss')
    ax[0, 0].set_ylabel('Cumprod Gain')
    ax[0, 0].legend()

    ax[0, 1].plot(np.arange(0, 0.1, 0.001), resulult_pick_winrate, label='Pick Winrate')
    ax[0, 1].plot(np.arange(0, 0.1, 0.001), resultst_day_winrate, label='Day Winrate')
    ax[0, 1].set_title('Winrate')
    ax[0, 1].set_xlabel('Stop Loss')
    ax[0, 1].set_ylabel('Winrate')
    ax[0, 1].legend()

    ax[1, 0].plot(np.arange(0, 0.1, 0.001), resultst_month_backset, label='Month Backset')
    ax[1, 0].set_title('Month Backset')
    ax[1, 0].set_xlabel('Stop Loss')
    ax[1, 0].set_ylabel('Month Backset')
    ax[1, 0].legend()

    ax[1, 1].axis('off')  # Hide the last subplot
    # save figure
    plt.tight_layout()
    plt.savefig(f"backtest/{chromosome_set}/sl_scope.png")