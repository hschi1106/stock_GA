import requests
import pandas as pd
import tqdm
from os import listdir
from os.path import isfile, join
from datetime import datetime, timedelta
import os


finmind_token = "YOUR_TOKEN"

def get_stock(start_date="2020-01-01", end_date="2020-01-01"):
    global finmind_token
    path_dayK = "./data/stock_dayK"
    path_dayK_files = [f.split('.')[0] for f in listdir(path_dayK) if isfile(join(path_dayK, f))]
    print(path_dayK_files)
    url = "https://api.finmindtrade.com/api/v4/data"
    for stock_id in tqdm.tqdm(path_dayK_files):
        parameter = {
            "dataset": "TaiwanStockPrice",
            "data_id": stock_id,
            "start_date": start_date,
            "end_date": end_date,
            "token": finmind_token,
        }

        resp = requests.get(url, params=parameter)
        data = resp.json()
        data = pd.DataFrame(data["data"])
        data.to_csv(f"./data/stock_dayK/{stock_id}.csv", index=False)
        print(f"{stock_id} done")

def get_one_dayK(date="2025-05-28"):
    global finmind_token
    url = "https://api.finmindtrade.com/api/v4/data"
    headers = {"Authorization": f"Bearer {finmind_token}"}
    parameter = {
        "dataset": "TaiwanStockPrice",
        "start_date": date,
    }
    resp = requests.get(url, headers=headers, params=parameter)
    data = resp.json()
    data = pd.DataFrame(data["data"])


    path_dayK = "./data/stock_dayK"
    path_dayK_files = [f.split('.')[0] for f in listdir(path_dayK) if isfile(join(path_dayK, f))]
    for stock_id in tqdm.tqdm(path_dayK_files):
        # insert one line to old datas
        df_dayK = pd.read_csv(f"{path_dayK}/{stock_id}.csv")
        if df_dayK.empty:
            continue
        try:
            df_dayK.loc[len(df_dayK)] = data[data['stock_id'] == stock_id].iloc[0]
            df_dayK = df_dayK.drop_duplicates(subset=['date'], keep='last')
            df_dayK.to_csv(f"{path_dayK}/{stock_id}.csv", index=False)
            print(f"{stock_id} done")
        except:
            print(f"{stock_id} error")
            continue

def get_index(start_date="2020-01-01", end_date="2020-01-01"):
    global finmind_token
    path_dayK = "./data/index_dayK"
    path_dayK_files = [f.split('.')[0] for f in listdir(path_dayK) if isfile(join(path_dayK, f))]
    print(path_dayK_files)
    url = "https://api.finmindtrade.com/api/v4/data"
    for index_id in tqdm.tqdm(path_dayK_files):
        parameter = {
            "dataset": "TaiwanStockPrice",
            "data_id": index_id,
            "start_date": start_date,
            "end_date": end_date,
            "token": finmind_token,
        }

        resp = requests.get(url, params=parameter)
        data = resp.json()
        data = pd.DataFrame(data["data"])
        data.to_csv(f"./data/index_dayK/{index_id}.csv", index=False)
        print(f"{index_id} done")
        
def get_futures_tick(futures_id, start_date="2020-01-01", end_date="2020-01-01"):
    global finmind_token

    path_futures_tick = f"./data/futures_tick/{futures_id}"
    os.makedirs(path_futures_tick, exist_ok=True)

    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    date_list = pd.date_range(start=start_date, end=end_date)
    for date in tqdm.tqdm(date_list, desc=f"Fetching {futures_id}"):
        date_str = date.strftime("%Y-%m-%d")
        url = "https://api.finmindtrade.com/api/v4/data"
        parameter = {
            "dataset": "TaiwanFuturesTick",
            "data_id": futures_id,
            "start_date": date_str,
            "token": finmind_token,
        }
        resp = requests.get(url, params=parameter)
        data = resp.json()
        data = pd.DataFrame(data["data"])
        save_path = os.path.join(path_futures_tick, f"{date_str}.csv")
        data.to_csv(save_path, index=False)
        current_date += timedelta(days=1)
        
    print(f"{futures_id} done")

def get_institution(start_date="2020-01-01", end_date="2020-01-01"):
    global finmind_token

    path_inst = f"./data/stock_institution/"
    os.makedirs(path_inst, exist_ok=True)

    path_dayK = "./data/stock_dayK"
    path_dayK_files = [f.split('.')[0] for f in listdir(path_dayK) if isfile(join(path_dayK, f))]
    print(path_dayK_files)

    url = "https://api.finmindtrade.com/api/v4/data"
    headers = {"Authorization": f"Bearer {finmind_token}"}
    for stock_id in tqdm.tqdm(path_dayK_files):
        parameter = {
            "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
            "data_id": stock_id,
            "start_date": start_date,
            "end_date": end_date,
        }
        data = requests.get(url, headers=headers, params=parameter)
        data = data.json()
        data = pd.DataFrame(data['data'])

        # -- front processing --

        # Pivot the data so each 'name' becomes a column with buy and sell values
        pivoted = data.pivot_table(
            index=['date', 'stock_id'],
            columns='name',
            values=['buy', 'sell'],
            aggfunc='sum',
            fill_value=0
        )

        # Flatten MultiIndex columns
        pivoted.columns = [f"{col[1]}_{col[0]}" for col in pivoted.columns]
        pivoted = pivoted.reset_index()

        # Optional: rename columns to match required output
        pivoted = pivoted.rename(columns={
            'Foreign_Investor_buy': 'fb',
            'Foreign_Investor_sell': 'fs',
            'Foreign_Dealer_Self_buy': 'fdb',
            'Foreign_Dealer_Self_sell': 'fds',
            'Investment_Trust_buy': 'ib',
            'Investment_Trust_sell': 'is',
            'Dealer_self_buy': 'db',
            'Dealer_self_sell': 'ds',
            'Dealer_Hedging_buy': 'dhb',
            'Dealer_Hedging_sell': 'dhs',
        })

        pivoted.to_csv(f"{path_inst}/{stock_id}.csv", index=False)
        print(f"{stock_id} done")

def get_institution_special(stock_id, start_date="2020-01-01", end_date="2020-01-01", save_path="./test.csv"):
    global finmind_token

    url = "https://api.finmindtrade.com/api/v4/data"
    headers = {"Authorization": f"Bearer {finmind_token}"}
    parameter = {
        "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
        "data_id": stock_id,
        "start_date": start_date,
        "end_date": end_date,
    }
    data = requests.get(url, headers=headers, params=parameter)
    data = data.json()
    data = pd.DataFrame(data['data'])

    # -- front processing --

    # Pivot the data so each 'name' becomes a column with buy and sell values
    pivoted = data.pivot_table(
        index=['date', 'stock_id'],
        columns='name',
        values=['buy', 'sell'],
        aggfunc='sum',
        fill_value=0
    )

    # Flatten MultiIndex columns
    pivoted.columns = [f"{col[1]}_{col[0]}" for col in pivoted.columns]
    pivoted = pivoted.reset_index()

    # Optional: rename columns to match required output
    pivoted = pivoted.rename(columns={
        'Foreign_Investor_buy': 'fb',
        'Foreign_Investor_sell': 'fs',
        'Foreign_Dealer_Self_buy': 'fdb',
        'Foreign_Dealer_Self_sell': 'fds',
        'Investment_Trust_buy': 'ib',
        'Investment_Trust_sell': 'is',
        'Dealer_self_buy': 'db',
        'Dealer_self_sell': 'ds',
        'Dealer_Hedging_buy': 'dhb',
        'Dealer_Hedging_sell': 'dhs',
    })

    pivoted.to_csv(f"{save_path}", index=False)
    print(f"{stock_id} done")

if __name__ == "__main__":
    # get_institution_special("1321", start_date="2023-08-29", end_date="2023-08-30")
    # get_stock(start_date="2020-01-01", end_date="2025-05-28")
    # get_institution(start_date="2020-01-01", end_date="2025-05-28")
    # get_one_dayK()
    # get_index()