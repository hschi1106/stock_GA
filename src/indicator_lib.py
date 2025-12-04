import pandas as pd
import numpy as np
import datetime
from numba import njit

def sma(data, period):
    new_data = np.zeros(len(data))
    for d in range(period, len(data)):
        new_data[d] = np.mean(data[d-period:d])
    return new_data

def std(data, period):
    new_data = np.zeros(len(data))
    for d in range(period, len(data)):
        new_data[d] = np.std(data[d-period:d])
    return new_data

def ema(data, period):
    alpha = 2 / (period + 1)
    new_data = np.zeros(len(data))
    new_data[0] = 1
    new_data[1] = data[0]
    for d in range(2, len(data)):
        new_data[d] = alpha * data[d-1] + (1 - alpha) * new_data[d-2]
    return new_data

def cal_DM(high, low):
    n = len(high)
    pos_DM = np.zeros(n)
    neg_DM = np.zeros(n)
    
    pos_DM[0] = 0
    neg_DM[0] = 0
    neg_DM[1] = 0
    pos_DM[1] = 0
    for i in range(2, n):
        up_move = high[i-1] - high[i-2]
        down_move = low[i-2] - low[i-1]
        
        if up_move > down_move and up_move > 0:
            pos_DM[i] = up_move
        else:
            pos_DM[i] = 0
        
        if down_move > up_move and down_move > 0:
            neg_DM[i] = down_move
        else:
            neg_DM[i] = 0
    return pos_DM, neg_DM

def cal_TR(high, low, close):
    n = len(high)
    TR = np.zeros(n)
    
    TR[0] = 0
    TR[1] = 0
    for i in range(2, n):
        TR[i] = max(high[i-1] - low[i-1],
                    abs(high[i-1] - close[i-2]),
                    abs(low[i-1] - close[i-2]))
    return TR

def cal_DI(high, low, close, period=14):
    n = len(high)
    pos_DM, neg_DM = cal_DM(high, low)
    tr = cal_TR(high, low, close)
    
    pos_DI = np.zeros(n)
    neg_DI = np.zeros(n)
    
    for i in range(period, n):
        sum_pos_DM = np.sum(pos_DM[i - period:i])
        sum_neg_DM = np.sum(neg_DM[i - period:i])
        sum_TR = np.sum(tr[i - period:i])
        if sum_TR != 0:
            pos_DI[i] = 100 * sum_pos_DM / sum_TR
            neg_DI[i] = 100 * sum_neg_DM / sum_TR
        else:
            pos_DI[i] = 0
            neg_DI[i] = 0
    return pos_DI, neg_DI

def cal_DX(pos_DI, neg_DI, period=14):
    n = len(pos_DI)
    DX = np.zeros(n)
    
    for i in range(period, n):
        denom = pos_DI[i-1] + neg_DI[i-1]
        if denom != 0:
            DX[i] = 100 * abs(pos_DI[i-1] - neg_DI[i-1]) / denom
        else:
            DX[i] = 0
    return DX

def cal_ADX(DX, period=14):
    n = len(DX)
    ADX = np.zeros(n)
    
    for i in range(2 * period - 1, n):
        ADX[i] = np.mean(DX[i - period:i])
    return ADX

def cal_MACD(close, short_period=12, long_period=26, signal_period=9):
    macd_line = ema(close, short_period) - ema(close, long_period)
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def cal_RSI(close, period=14):
    n = len(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    for i in range(1, n):
        change = close[i] - close[i-1]
        if change > 0:
            gain[i] = change
            loss[i] = 0
        else:
            gain[i] = 0
            loss[i] = abs(change)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    for i in range(period, n):
        avg_gain[i] = np.mean(gain[i - period:i])
        avg_loss[i] = np.mean(loss[i - period:i])
    
    rs = np.divide(
        avg_gain,
        avg_loss,
        out=np.zeros_like(avg_gain),
        where=avg_loss != 0
    )
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def cal_KDJ(close, high, low, period=14):
    n = len(close)
    rsv = np.zeros(n)
    k = np.zeros(n)
    d = np.zeros(n)
    
    for i in range(period, n):
        min_low = np.min(low[i - period:i])
        max_high = np.max(high[i - period:i])
        rsv[i] = (close[i-1] - min_low) / (max_high - min_low) * 100
    
    for i in range(period, n):
        if i == period:
            k[i] = rsv[i]
            d[i] = rsv[i]
        else:
            k[i] = (2 / 3) * k[i - 1] + (1 / 3) * rsv[i]
            d[i] = (2 / 3) * d[i - 1] + (1 / 3) * k[i]
    
    j = 3 * k - 2 * d
    
    return k, d, j

def indicator_filter(df, chromosome): # fitness function
    mask = np.ones(len(df), dtype=bool)

    for idx, gene in enumerate(chromosome):
        if gene == 1:
            mask &= df[f"idc{idx}"].values
        elif gene == -1:
            mask &= ~df[f"idc{idx}"].values

        if not mask.any():
            return df.iloc[0:0]

    return df.loc[mask]

def gen_condition_list(seed=42):
    all_conditions = []

    # [cond1, cond2, multiplier] => cond1 > cond2 * multiplier
    price_columns = ['open', 'sma5', 'sma20', 'sma60', 'upper', 'lower', 'prevc']
    for fpc in price_columns:
        for spc in price_columns:
            if fpc == spc:
                break
            all_conditions.append([fpc, spc, 1])
    
    # foreign, dealer, investment
    foreign_columns = ['foreign', 'dealer', 'investment']
    for fpc in foreign_columns:
        all_conditions.append([fpc, 'one', -0.1])
        all_conditions.append([fpc, 'one', -0.05])
        all_conditions.append([fpc, 'one', 0])
        all_conditions.append([fpc, 'one', 0.05])
        all_conditions.append([fpc, 'one', 0.1])
    
    # remove 'upper' and 'lower' pair
    all_conditions = [cond for cond in all_conditions if not (cond[0] == 'upper' and cond[1] == 'lower') and not (cond[0] == 'lower' and cond[1] == 'upper')]
    
    all_conditions.append(['width', 'one', 0.06])
    all_conditions.append(['width', 'one', 0.12])

    all_conditions.append(['vol_upper', 'vol', 1])
    all_conditions.append(['vol_lower', 'vol', 1])
    all_conditions.append(['vol_sma', 'vol', 1])

    all_conditions.append(['prevgain', 'one', 0.01])
    all_conditions.append(['prevgain', 'one', 0])
    all_conditions.append(['prevgain', 'one', -0.01])

    all_conditions.append(['prevgain2', 'one', 0.01])
    all_conditions.append(['prevgain2', 'one', 0])
    all_conditions.append(['prevgain2', 'one', -0.01])

    all_conditions.append(['prevgain3', 'one', 0.01])
    all_conditions.append(['prevgain3', 'one', 0])
    all_conditions.append(['prevgain3', 'one', -0.01])

    all_conditions.append(['prevgain4', 'one', 0])
    all_conditions.append(['prevgain5', 'one', 0])

    all_conditions.append(['prevgain', 'prevgain2', 1])

    all_conditions.append(['ADX', 'one', 20])
    all_conditions.append(['ADX', 'one', 40])

    all_conditions.append(['vol', 'prevvol', 1])
    all_conditions.append(['vol', 'prevvol', 1.5])
    all_conditions.append(['vol', 'prevvol', 2])
    all_conditions.append(['vol', 'prevvol', 3])

    all_conditions.append(['vol_money', 'one', 5e7])
    all_conditions.append(['vol_money', 'one', 1e8])
    all_conditions.append(['vol_money', 'one', 5e8])
    all_conditions.append(['vol_money', 'one', 1e9])

    if seed is not None:
        np.random.seed(seed)
        np.random.shuffle(all_conditions)

    return all_conditions

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
        # mask = pd.to_datetime(df['date'], errors='coerce') >= pd.Timestamp('2022-01-01')
        # df = df.loc[mask]
            
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

def process_dataframe_future(df, train=False, condition_list=None):
    df = df.copy()
    df.reset_index(drop=True, inplace=True)
    
    # print(df.columns)
    
    # check it is a valid dataframe
    if df.empty:
        return df
    if not all(col in df.columns for col in ['time', 'open', 'high', 'low', 'close', 'volume']):
        raise ValueError("DataFrame does not contain required columns")

    # basic cleanup
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.dropna(inplace=True)

    df['gain'] = (df['close'] - df['open']) / df['open'] - 0.00015
    df['up_threshold'] = df['open'] * 1.002
    df['down_threshold'] = df['open'] * 0.998
    
    signals = []
    for i in range(len(df)):
        up = df['up_threshold'][i]
        down = df['down_threshold'][i]
        signal = 0
        for j in range(i, len(df)):
            hit_up   = df.at[j, 'high'] >= up
            hit_down = df.at[j, 'low']  <= down
            
            if hit_up and hit_down:
                signal = 0
                break
            elif hit_up:
                signal =  1
                break
            elif hit_down:
                signal = -1
                break
        signals.append(signal)
        
    df['exit_signal'] = signals
    df.drop(columns=['up_threshold', 'down_threshold'], inplace=True)
    
    df['open_1'] = df['open'].shift(1)
    df['high_1'] = df['high'].shift(1)
    df['low_1'] = df['low'].shift(1)
    df['close_1'] = df['close'].shift(1)
    df['volume_1'] = df['volume'].shift(1)
    
    df['open_2'] = df['open'].shift(2)
    df['high_2'] = df['high'].shift(2)
    df['low_2'] = df['low'].shift(2)
    df['close_2'] = df['close'].shift(2)
    df['volume_2'] = df['volume'].shift(2)
    
    df['open_3'] = df['open'].shift(3)
    df['high_3'] = df['high'].shift(3)
    df['low_3'] = df['low'].shift(3)
    df['close_3'] = df['close'].shift(3)
    df['volume_3'] = df['volume'].shift(3)
    
    # calculate indicators
    close_array = np.array(df['close'].to_list())
    # high_array = np.array(df['high'].to_list())
    # low_array = np.array(df['low'].to_list())
    std_20 = std(close_array, 20)
    ema_20 = ema(close_array, 20)
    upper = ema_20 + 2 * std_20
    lower = ema_20 - 2 * std_20
    
    # BBands
    df['width'] = (upper - lower) / ema_20
    df['upper'] = upper
    df['lower'] = lower
    
    # smas
    df['sma5'] = sma(close_array, 5)
    df['sma10'] = sma(close_array, 10)
    df['sma20'] = sma(close_array, 20)

    df['one'] = 1
    
    # df = df[60:] # drop front 60 days

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
        # mask = pd.to_datetime(df['date'], errors='coerce') >= pd.Timestamp('2022-01-01')
        # df = df.loc[mask]
            
        # remain only gain and indicators
        remain_list = [f'idc{i}' for i in range(idc_cnt)]
        remain_list.append('gain')
        remain_list.append('exit_signal')
        df = df[remain_list]
    
    # final cleanup
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def gen_condition_list_future(seed=42):
    all_conditions = []
    
    all_conditions.append(['width', 'one', 0.03])
    all_conditions.append(['width', 'one', 0.06])
    # all_conditions.append(['width', 'one', 0.09])
    # all_conditions.append(['width', 'one', 0.12])
    # all_conditions.append(['width', 'one', 0.15])

    # [cond1, cond2, multiplier] => cond1 > cond2 * multiplier
    price_columns = [
        # 'open_1', 'open_2', 'open_3', 
        'close_1', 'close_2', 'close_3', 
        # 'high_1', 'high_2', 'high_3', 
        # 'low_1', 'low_2', 'low_3',
        'upper', 'lower',
        'sma5', 'sma10', 'sma20',
    ]
    for fpc in price_columns:
        for spc in price_columns:
            if fpc == spc:
                break
            all_conditions.append([fpc, spc, 1])
            
    volume_columns = ['volume_1', 'volume_2', 'volume_3']
    for fpc in volume_columns:
        for spc in volume_columns:
            if fpc == spc:
                break
            # all_conditions.append([fpc, spc, 0.1])
            all_conditions.append([fpc, spc, 0.2])
            all_conditions.append([fpc, spc, 0.5])
            all_conditions.append([fpc, spc, 1])
            all_conditions.append([fpc, spc, 2])
            all_conditions.append([fpc, spc, 5])
            # all_conditions.append([fpc, spc, 10])
    
    # remove high_n and low_n pair
    all_conditions = [cond for cond in all_conditions if not (cond[0] == 'high_1' and cond[1] == 'low_1') and not (cond[0] == 'low_1' and cond[1] == 'high_1')]
    all_conditions = [cond for cond in all_conditions if not (cond[0] == 'high_2' and cond[1] == 'low_2') and not (cond[0] == 'low_2' and cond[1] == 'high_2')]
    all_conditions = [cond for cond in all_conditions if not (cond[0] == 'high_3' and cond[1] == 'low_3') and not (cond[0] == 'low_3' and cond[1] == 'high_3')]

    if seed is not None:
        np.random.seed(seed)
        np.random.shuffle(all_conditions)

    return all_conditions

@njit(cache=True)
def compute_upi_jit(returns):
    cum_returns = np.empty(len(returns) + 1, dtype=np.float64)
    cum_returns[0] = 1.0
    for i in range(len(returns)):
        cum_returns[i + 1] = cum_returns[i] * returns[i]

    rolling_max = np.empty_like(cum_returns)
    rolling_max[0] = cum_returns[0]
    for i in range(1, len(cum_returns)):
        rolling_max[i] = max(rolling_max[i - 1], cum_returns[i])

    drawdowns = np.maximum(0, (rolling_max - cum_returns) / rolling_max)
    p = 2
    ulcer_index = (np.mean(drawdowns**p))**(1.0/p)

    if ulcer_index == 0:
        return 0

    total_return = cum_returns[-1] / cum_returns[0] - 1
    return total_return / ulcer_index