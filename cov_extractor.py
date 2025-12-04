import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
from src.indicator_lib import sma, std, ema, cal_DM, cal_TR, cal_DI, cal_DX, cal_ADX, cal_RSI, cal_MACD, cal_KDJ
from src.utils import load_files
import json
import os

def process_dataframe(df, train=False, condition_list=None):
    df = df.copy()
    
    # check it is a valid dataframe
    if df.empty:
        return df
    if not all(col in df.columns for col in ['date', 'stock_id', 'Trading_Volume', 'Trading_money', 'open', 'max', 'min', 'close', 'spread', 'Trading_turnover']):
        raise ValueError("DataFrame does not contain required columns")

    # basic cleanup
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.dropna(inplace=True)

    # calculate indicators
    close_array = np.array(df['close'].to_list())
    high_array = np.array(df['max'].to_list())
    low_array = np.array(df['min'].to_list())
    std_20 = std(close_array, 20)
    ema_20 = ema(close_array, 20)
    upper = ema_20 + 2 * std_20
    lower = ema_20 - 2 * std_20

    # BBands
    df['width'] = (upper - lower) / ema_20
    df['width1'] = df['width'].shift(1)
    df['width2'] = df['width'].shift(2)
    df['upper'] = upper
    df['upper1'] = df['upper'].shift(1)
    df['upper2'] = df['upper'].shift(2)
    df['lower'] = lower
    df['lower1'] = df['lower'].shift(1)
    df['lower2'] = df['lower'].shift(2)

    # smas
    df['sma5'] = sma(close_array, 5)
    df['sma5_1'] = df['sma5'].shift(1)
    df['sma5_2'] = df['sma5'].shift(2)
    df['sma20'] = sma(close_array, 21)
    df['sma20_1'] = df['sma20'].shift(1)
    df['sma20_2'] = df['sma20'].shift(2)
    df['sma60'] = sma(close_array, 60)
    df['sma60_1'] = df['sma60'].shift(1)
    df['sma60_2'] = df['sma60'].shift(2)

    # ADX
    pos_DI, neg_DI = cal_DI(high_array, low_array, close_array)
    DX = cal_DX(pos_DI, neg_DI)
    ADX = cal_ADX(DX)
    df['ADX'] = ADX
    df['ADX1'] = df['ADX'].shift(1)
    df['ADX2'] = df['ADX'].shift(2)
    
    df['rsi'] = cal_RSI(close_array, 14)
    
    macd_line, signal_line, histogram = cal_MACD(close_array, 12, 26, 9)
    df['macd'] = macd_line
    df['macd1'] = df['macd'].shift(1)
    df['signal'] = signal_line
    df['signal1'] = df['signal'].shift(1)
    df['histogram'] = histogram
    df['histogram1'] = df['histogram'].shift(1)
    
    k, d, j = cal_KDJ(close_array, high_array, low_array, 14)
    df['k'] = k
    df['k1'] = df['k'].shift(1)
    df['d'] = d
    df['d1'] = df['d'].shift(1)
    df['j'] = j
    df['j1'] = df['j'].shift(1)
    
    # previous values
    df['prevc'] = df['close'] - df['spread']
    df['prevc2'] = df['prevc'].shift(1)
    df['prevc3'] = df['prevc'].shift(2)
    df['prevc4'] = df['prevc'].shift(3)
    df['prevc5'] = df['prevc'].shift(4)

    df['prevo'] = df['open'].shift(1)
    df['prevh'] = df['max'].shift(1)
    df['prevl'] = df['min'].shift(1)
    
    # volume
    vol_array = np.array(df['Trading_Volume'].to_list())
    vol_sma = sma(vol_array, 20)
    vol_std = std(vol_array, 20)
    df['vol'] = df['Trading_Volume'].shift(1)
    df['vol_money'] = df['Trading_money'].shift(1)
    df['prevvol'] = df['vol'].shift(1)
    df['prevvol2'] = df['vol'].shift(2)
    df['vol_sma'] = vol_sma
    df['vol_upper'] = vol_sma + 2 * vol_std
    df['vol_lower'] = vol_sma - 2 * vol_std

    # gain
    df['gain'] = (df['close'] - df['open']) / df['open']
    df['prevgain'] = df['gain'].shift(1)
    df['prevgain2'] = df['gain'].shift(2)
    df['prevgain3'] = df['gain'].shift(3)
    df['prevgain4'] = df['gain'].shift(4)
    df['prevgain5'] = df['gain'].shift(5)

    # foreign, dealer, investment
    df['dealer_net_rate'] = (df['db'] + df['dhb'] - df['ds'] - df['dhs']) / df['Trading_Volume']
    df['foreign_net_rate'] = (df['fb'] + df['fdb'] - df['fs'] - df['fds']) / df['Trading_Volume']
    df['investment_net_rate'] = (df['ib'] - df['is']) / df['Trading_Volume']

    df['dealer'] = df['dealer_net_rate'].shift(1)
    df['foreign'] = df['foreign_net_rate'].shift(1)
    df['investment'] = df['investment_net_rate'].shift(1)

    df['one'] = 1
    
    df = df[60:] # drop front 60 days

    idc_cnt = len(condition_list)
    new_cols = {}
    for idx, tech_idc in enumerate(condition_list):
        if len(tech_idc) == 3:
            if type(tech_idc[0]) == str and type(tech_idc[1]) == str and (type(tech_idc[2]) == float or type(tech_idc[2]) == int):
                # df[f'idc{idx}'] = df[tech_idc[0]] > df[tech_idc[1]] * tech_idc[2]
                new_cols[f'idc{idx}'] = df[tech_idc[0]] > df[tech_idc[1]] * tech_idc[2]
            else:
                raise ValueError("Invalid condition list format (type error)")
        else:
            raise ValueError("Invalid condition list format (len != 3)")
    
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    if train:
        # drop date before 2023-01-01
        mask = pd.to_datetime(df['date'], errors='coerce') >= pd.Timestamp('2022-01-01')
        df = df.loc[mask]
            
        # remain only gain and indicators
        remain_list = [f'idc{i}' for i in range(idc_cnt)]
        remain_list.append('gain')
        remain_list.append('date')
        remain_list.append('vol')
        remain_list.append('prevc')
        df = df[remain_list]
    
    # final cleanup
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def gen_condition_list(seed=42):
    all_conditions = []

    # [cond1, cond2, multiplier] => cond1 > cond2 * multiplier
    price_columns = ['open', 'sma5', 'sma20', 'sma60', 'upper', 'lower', 'prevc', 'sma5_1', 'sma5_2', 'sma20_1', 'sma20_2', 'sma60_1', 'sma60_2', 'prevc2', 'prevc3', 'prevc4', 'prevc5']
    for fpc in price_columns:
        for spc in price_columns:
            if fpc == spc:
                break
            all_conditions.append([fpc, spc, 1])
    
    
    # foreign, dealer, investment
    foreign_columns = ['foreign', 'dealer', 'investment']
    for fpc in foreign_columns:
        all_conditions.append([fpc, 'one', -0.5])
        all_conditions.append([fpc, 'one', -0.4])
        all_conditions.append([fpc, 'one', -0.3])
        all_conditions.append([fpc, 'one', -0.2])
        all_conditions.append([fpc, 'one', -0.1])
        all_conditions.append([fpc, 'one', -0.08])
        all_conditions.append([fpc, 'one', -0.06])
        all_conditions.append([fpc, 'one', -0.04])
        all_conditions.append([fpc, 'one', -0.02])
        all_conditions.append([fpc, 'one', 0])
        all_conditions.append([fpc, 'one', 0.02])
        all_conditions.append([fpc, 'one', 0.04])
        all_conditions.append([fpc, 'one', 0.06])
        all_conditions.append([fpc, 'one', 0.08])
        all_conditions.append([fpc, 'one', 0.1])
        all_conditions.append([fpc, 'one', 0.2])
        all_conditions.append([fpc, 'one', 0.3])
        all_conditions.append([fpc, 'one', 0.4])
        all_conditions.append([fpc, 'one', 0.5])

    # remove 'upper' and 'lower' pair
    all_conditions = [cond for cond in all_conditions if not (cond[0] == 'upper' and cond[1] == 'lower') and not (cond[0] == 'lower' and cond[1] == 'upper')]
    
    all_conditions.append(['width', 'one', 0.03])
    all_conditions.append(['width', 'one', 0.06])
    all_conditions.append(['width', 'one', 0.09])
    all_conditions.append(['width', 'one', 0.12])
    all_conditions.append(['width', 'one', 0.15])
    all_conditions.append(['width', 'one', 0.18])
    
    all_conditions.append(['width', 'width1', 1])
    all_conditions.append(['width', 'width2', 1])
    all_conditions.append(['width1', 'width2', 1])
    all_conditions.append(['upper', 'upper1', 1])
    all_conditions.append(['upper', 'upper2', 1])
    all_conditions.append(['upper1', 'upper2', 1])
    all_conditions.append(['lower', 'lower1', 1])
    all_conditions.append(['lower', 'lower2', 1])
    all_conditions.append(['lower1', 'lower2', 1])
    
    all_conditions.append(['vol_upper', 'one', 0.5])
    all_conditions.append(['vol_upper', 'vol', 0.67])
    all_conditions.append(['vol_upper', 'vol', 1])
    all_conditions.append(['vol_upper', 'vol', 1.5])
    all_conditions.append(['vol_upper', 'vol', 2])
    
    all_conditions.append(['vol_lower', 'one', 0.5])
    all_conditions.append(['vol_lower', 'vol', 0.67])
    all_conditions.append(['vol_lower', 'vol', 1])
    all_conditions.append(['vol_lower', 'vol', 1.5])
    all_conditions.append(['vol_lower', 'vol', 2])
    
    all_conditions.append(['vol_sma', 'one', 0.5])
    all_conditions.append(['vol_sma', 'vol', 0.67])
    all_conditions.append(['vol_sma', 'vol', 1])
    all_conditions.append(['vol_sma', 'vol', 1.5])
    all_conditions.append(['vol_sma', 'vol', 2])

    all_conditions.append(['prevgain', 'one', 0.09])
    all_conditions.append(['prevgain', 'one', 0.07])
    all_conditions.append(['prevgain', 'one', 0.05])
    all_conditions.append(['prevgain', 'one', 0.03])
    all_conditions.append(['prevgain', 'one', 0.01])
    all_conditions.append(['prevgain', 'one', 0])
    all_conditions.append(['prevgain', 'one', -0.01])
    all_conditions.append(['prevgain', 'one', -0.03])
    all_conditions.append(['prevgain', 'one', -0.05])
    all_conditions.append(['prevgain', 'one', -0.07])
    all_conditions.append(['prevgain', 'one', -0.09])

    all_conditions.append(['prevgain2', 'one', 0.09])
    all_conditions.append(['prevgain2', 'one', 0.07])
    all_conditions.append(['prevgain2', 'one', 0.05])
    all_conditions.append(['prevgain2', 'one', 0.03])
    all_conditions.append(['prevgain2', 'one', 0.01])
    all_conditions.append(['prevgain2', 'one', 0])
    all_conditions.append(['prevgain2', 'one', -0.01])
    all_conditions.append(['prevgain2', 'one', -0.03])
    all_conditions.append(['prevgain2', 'one', -0.05])
    all_conditions.append(['prevgain2', 'one', -0.07])
    all_conditions.append(['prevgain2', 'one', -0.09])
    
    all_conditions.append(['prevgain3', 'one', 0.09])
    all_conditions.append(['prevgain3', 'one', 0.07])
    all_conditions.append(['prevgain3', 'one', 0.05])
    all_conditions.append(['prevgain3', 'one', 0.03])
    all_conditions.append(['prevgain3', 'one', 0.01])
    all_conditions.append(['prevgain3', 'one', 0])
    all_conditions.append(['prevgain3', 'one', -0.01])
    all_conditions.append(['prevgain3', 'one', -0.03])
    all_conditions.append(['prevgain3', 'one', -0.05])
    all_conditions.append(['prevgain3', 'one', -0.07])
    all_conditions.append(['prevgain3', 'one', -0.09])
    
    all_conditions.append(['prevgain4', 'one', 0.09])
    all_conditions.append(['prevgain4', 'one', 0.07])
    all_conditions.append(['prevgain4', 'one', 0.05])
    all_conditions.append(['prevgain4', 'one', 0.03])
    all_conditions.append(['prevgain4', 'one', 0.01])
    all_conditions.append(['prevgain4', 'one', 0])
    all_conditions.append(['prevgain4', 'one', -0.01])
    all_conditions.append(['prevgain4', 'one', -0.03])
    all_conditions.append(['prevgain4', 'one', -0.05])
    all_conditions.append(['prevgain4', 'one', -0.07])
    all_conditions.append(['prevgain4', 'one', -0.09])
    
    all_conditions.append(['prevgain5', 'one', 0.09])
    all_conditions.append(['prevgain5', 'one', 0.07])
    all_conditions.append(['prevgain5', 'one', 0.05])
    all_conditions.append(['prevgain5', 'one', 0.03])
    all_conditions.append(['prevgain5', 'one', 0.01])
    all_conditions.append(['prevgain5', 'one', 0])
    all_conditions.append(['prevgain5', 'one', -0.01])
    all_conditions.append(['prevgain5', 'one', -0.03])
    all_conditions.append(['prevgain5', 'one', -0.05])
    all_conditions.append(['prevgain5', 'one', -0.07])
    all_conditions.append(['prevgain5', 'one', -0.09])

    all_conditions.append(['prevgain', 'prevgain2', 1])
    all_conditions.append(['prevgain', 'prevgain3', 1])
    all_conditions.append(['prevgain', 'prevgain4', 1])
    all_conditions.append(['prevgain', 'prevgain5', 1])
    all_conditions.append(['prevgain2', 'prevgain3', 1])
    all_conditions.append(['prevgain2', 'prevgain4', 1])
    all_conditions.append(['prevgain2', 'prevgain5', 1])
    all_conditions.append(['prevgain3', 'prevgain4', 1])
    all_conditions.append(['prevgain3', 'prevgain5', 1])
    all_conditions.append(['prevgain4', 'prevgain5', 1])

    all_conditions.append(['ADX', 'one', 5])
    all_conditions.append(['ADX', 'one', 10])
    all_conditions.append(['ADX', 'one', 15])
    all_conditions.append(['ADX', 'one', 20])
    all_conditions.append(['ADX', 'one', 25])
    all_conditions.append(['ADX', 'one', 30])
    all_conditions.append(['ADX', 'one', 35])
    all_conditions.append(['ADX', 'one', 40])
    all_conditions.append(['ADX', 'one', 45])
    all_conditions.append(['ADX', 'one', 50])
    all_conditions.append(['ADX', 'one', 55])
    all_conditions.append(['ADX', 'one', 60])
    
    all_conditions.append(['ADX', 'ADX1', 1])
    all_conditions.append(['ADX', 'ADX2', 1])
    all_conditions.append(['ADX1', 'ADX2', 1])

    all_conditions.append(['vol', 'prevvol', 0.33])
    all_conditions.append(['vol', 'prevvol', 0.5])
    all_conditions.append(['vol', 'prevvol', 0.67])
    all_conditions.append(['vol', 'prevvol', 1])
    all_conditions.append(['vol', 'prevvol', 1.5])
    all_conditions.append(['vol', 'prevvol', 2])
    all_conditions.append(['vol', 'prevvol', 3])
    
    all_conditions.append(['vol', 'prevvol2', 0.33])
    all_conditions.append(['vol', 'prevvol2', 0.5])
    all_conditions.append(['vol', 'prevvol2', 0.67])
    all_conditions.append(['vol', 'prevvol2', 1])
    all_conditions.append(['vol', 'prevvol2', 1.5])
    all_conditions.append(['vol', 'prevvol2', 2])
    all_conditions.append(['vol', 'prevvol2', 3])
    
    all_conditions.append(['prevvol', 'prevvol2', 0.33])
    all_conditions.append(['prevvol', 'prevvol2', 0.5])
    all_conditions.append(['prevvol', 'prevvol2', 0.67])
    all_conditions.append(['prevvol', 'prevvol2', 1])
    all_conditions.append(['prevvol', 'prevvol2', 1.5])
    all_conditions.append(['prevvol', 'prevvol2', 2])
    all_conditions.append(['prevvol', 'prevvol2', 3])
    
    all_conditions.append(['macd', 'signal', 1])
    all_conditions.append(['macd', 'signal1', 1])
    all_conditions.append(['macd1', 'signal', 1])
    all_conditions.append(['macd1', 'signal1', 1])
    all_conditions.append(['histogram', 'one', 0])
    all_conditions.append(['histogram1', 'one', 0])

    all_conditions.append(['k', 'd', 1])
    all_conditions.append(['k', 'd1', 1])
    all_conditions.append(['k1', 'd', 1])
    all_conditions.append(['k1', 'd1', 1])
    all_conditions.append(['k', 'j', 1])
    all_conditions.append(['k', 'j1', 1])
    all_conditions.append(['k1', 'j', 1])
    all_conditions.append(['k1', 'j1', 1])
    all_conditions.append(['d', 'j', 1])
    all_conditions.append(['d', 'j1', 1])
    all_conditions.append(['d1', 'j', 1])
    all_conditions.append(['d1', 'j1', 1])
    
    all_conditions.append(['rsi', 'one', 20])
    all_conditions.append(['rsi', 'one', 30])
    all_conditions.append(['rsi', 'one', 40])
    all_conditions.append(['rsi', 'one', 50])
    all_conditions.append(['rsi', 'one', 60])
    all_conditions.append(['rsi', 'one', 70])
    all_conditions.append(['rsi', 'one', 80])

    if seed is not None:
        np.random.seed(seed)
        np.random.shuffle(all_conditions)

    return all_conditions

def top_covariances(cond_list, stock_df, comb_size=1):
    covs, idxs = [], []
    
    for combo in tqdm(itertools.combinations(range(len(cond_list)), comb_size)):
        s = sum(stock_df.loc[:, f'idc{i}'].astype(int) for i in combo)
        cov = np.cov(s, stock_df.loc[:, 'gain'].astype(float))
        covs.append(cov[0, 1])
        idxs.append(combo)
        
    covs = np.array(covs)
    idxs = np.array(idxs)
        
    return covs, idxs
    

if __name__ == "__main__":
    direction = False # True for long, False for short
    discount_rate = 0.3 # 70% discount on transaction fee
    cond_list_seed = 123
    extract_num = 80

    # condition list
    cond_list = gen_condition_list(seed=cond_list_seed)
    idc_cnt = len(cond_list)
    print("Conditions:\n", cond_list, "\nCondition count:", idc_cnt)

    # file preprocessing
    path_dayK_files, _ = load_files(left_range=1)
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

    # filter stock data
    # stock_data_pds = stock_data_pds[stock_data_pds['prevc'] < 100]
    stock_data_pds = stock_data_pds[stock_data_pds['vol'] > 1000000]
    
    all_covs, all_idxs = [], []
    for size in (1, 2):
        covs, idxs = top_covariances(cond_list, stock_data_pds, comb_size=size)
        all_covs.append(covs)
        all_idxs.append(idxs)

    items = []
    for covs, idxs in zip(all_covs, all_idxs):
        for cov, combo in zip(covs, idxs):
            items.append((cov, tuple(combo)))
            
    items_sorted = sorted(items, key=lambda x: abs(x[0]), reverse=True)
    
    selected = []
    seen = set()
    for cov, combo in items_sorted:
        for idx in combo:
            cond = cond_list[idx]
            key = tuple(cond)
            if key in seen:
                break
            
            seen.add(key)
            selected.append(cond)
            
            if len(selected) >= extract_num:
                break
        if len(selected) >= extract_num:
            break

    with open(f'top{extract_num}_t_condlist.json', 'w', encoding='utf-8') as f:
        f.write('[\n')
        for i, cond in enumerate(selected):
            line = json.dumps(cond, ensure_ascii=False)
            if i < len(selected) - 1:
                f.write(f"{line}, \n")
            else:
                f.write(f"{line}\n")
        f.write(']')

    print(f"{len(selected)} highest covariance cond_list save to cov_condlist.json")
