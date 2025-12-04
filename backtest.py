import numpy as np
import pandas as pd
from tqdm import tqdm

import os, json

from src.indicator_lib import process_dataframe, indicator_filter, gen_condition_list
from src.utils import load_files

from matplotlib import pyplot as plt

if __name__ == "__main__":
    # CONSTANTS
    leverage = 1
    direction = False # True for long, False for short
    discount_rate = 0.3 # 70% discount on transaction fee
    stop_loss = 100 # 4.5% stop loss
    chromosome_set = 'new_espx'
    file_load_seed = 678

    cond_list = json.load(open(f"chromosomes/{chromosome_set}_cond.json", "r"))
    idc_cnt = len(cond_list)

    _, path_dayK_files = load_files(left_range=0.3, seed=file_load_seed)
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

    print(len(stock_data_pds))

    best_chromosome = json.load(open(f"chromosomes/{chromosome_set}.json", "r"))

    existed_chs = []
    for idxx, ch in enumerate(best_chromosome[:25]):
        if tuple(ch) in existed_chs:
            print(f"Chromosome {idxx} already exists, skipping...")
            continue
        existed_chs.append(tuple(ch))

        filtered_df = indicator_filter(stock_data_pds, ch)
        filtered_df = filtered_df.copy()

        filtered_df = filtered_df[filtered_df['prevc'] < 100]
        filtered_df = filtered_df[filtered_df['vol'] > 1000000]

        cx = len(filtered_df[filtered_df['gain'] < -0.096]) / len(filtered_df)

        corr_stand = filtered_df['gain'].corr(filtered_df['gain'])
        corr_pg = filtered_df['gain'].corr(filtered_df['prevgain'])
        corr_pg2 = filtered_df['gain'].corr(filtered_df['prevgain2'])
        corr_c = filtered_df['gain'].corr(filtered_df['prevc'])
        corr_vol = filtered_df['gain'].corr(filtered_df['vol_money'])

        # # filter prevc < 100
        # filtered_df = filtered_df[filtered_df['prevc'] < 100]
        # # filter vol > 1000 000 (1000 lot)
        # filtered_df = filtered_df[filtered_df['vol'] < 1000000]

        
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
        day_tags = []

        # monthly statistics
        month_tags = []
        month_cumprod_gains = []
        
        mean = np.mean(filtered_df['gain'])
        std = np.std(filtered_df['gain'])

        # group by month (YYYY-MM)
        filtered_df['date_m'] = filtered_df['date'].apply(lambda x: x[:7])
        grouped_month_df = filtered_df.groupby('date_m')
        for month_date, month_df in grouped_month_df:
            # for every month
            print(month_date)
            month_tags.append(month_date)

            # group by date
            grouped_day_df = month_df.groupby('date')
            one_month_gains = []
            for day_date, day_df in grouped_day_df:
                print(f"=== {day_date} ===")
                print(day_df[['stock_id', 'gain', 'prevgain', 'open']].to_string(index=False))
                
                day_mean = day_df['gain'].mean()
                one_month_gains.append(day_mean)
                day_gains.append(day_mean)
                day_tags.append(day_date)
                day_trys += 1
                day_wins += 1 if day_mean > 0 else 0
                print(f"{day_date}, mean: {day_mean}")
            
            month_cumprod_gain = (1 + np.array(one_month_gains)).cumprod()[-1]
            month_cumprod_gains.append(month_cumprod_gain - 1)

        pick_wins = sum([1 for gain in filtered_df['gain'] if gain > 0])
        pick_trys = len(filtered_df['gain'])
        
        print(f"Backtest Result:")
        array_gain = [100]
        for gain in month_cumprod_gains:
            array_gain.append(array_gain[-1] * (1 + gain))
            print(f"{month_tags[len(array_gain)-2]}, gain: {gain * 100}%, total: {array_gain[-1] - 100}%")
        
        array_day_gain = [100]
        for gain in day_gains:
            array_day_gain.append(array_day_gain[-1] * (1 + gain))
            # print(f"{day_tags[len(array_day_gain)-2]}, gain: {gain * 100}%, total: {array_day_gain[-1] - 100}%")

        print(f"Daily Try Count: {day_trys}")
        if day_trys != 0:
            print(f"Daily Win Rate: {day_wins / day_trys}")
        if pick_trys != 0:
            print(f"Pick Win Rate: {pick_wins / pick_trys}")
        print(f"Global Cumprod Gain: {array_gain[-1] - 100}%")
        print(f"Mean Gain: {mean * 100}%")
        print(f"Std Gain: {std * 100}%")
        print(f"Max Gain: {np.max(filtered_df['gain']) * 100}%")
        print(f"Min Gain: {np.min(filtered_df['gain']) * 100}%")
        print(f"Correlation with gain: {corr_stand}")
        print(f"Correlation with prevgain: {corr_pg}")
        print(f"Correlation with prevgain2: {corr_pg2}")
        print(f"Correlation with prevc: {corr_c}")
        print(f"Correlation with vol: {corr_vol}")

        print(f"Cxnt: {cx}")
        

        plt.figure(figsize=(15, 7))
        if day_trys != 0:
            plt.title("Backtest, win rate: {:.2f}, mean daily gain: {:.7f}".format(day_wins/day_trys, np.mean(day_gains)))
        
        # plot monthly gain
        plt.bar(month_tags, month_cumprod_gains)
        plt.xticks(rotation=90)
        os.makedirs(f"backtest/{chromosome_set}", exist_ok=True)
        plt.savefig(f"backtest/{chromosome_set}/monthly_gain_fig_{idxx}.png")
        plt.clf()
        
        # plot daily gain
        plt.figure(figsize=(15, 7))
        plt.title(f"Daily Gain , act count: {len(day_gains)}, pick win rate: {pick_wins/pick_trys if pick_trys != 0 else 0:.2f}")
        plt.bar(day_tags, day_gains)
        plt.xticks(rotation=90)
        plt.savefig(f"backtest/{chromosome_set}/daily_gain_fig_{idxx}.png")
        plt.clf()

        # plot cumprod
        plt.figure(figsize=(15, 7))
        plt.title("Cumulative Gain")
        plt.plot(array_day_gain, label="Daily Cumulative Gain")

        plt.ylabel("Cumulative Gain (%)")
        plt.savefig(f"backtest/{chromosome_set}/cumulative_gain_fig_{idxx}.png")
        plt.clf()

        # input("Press Enter to continue...")
