# ===== CELL 12 =====
"""Contains methods and classes to collect data from
Yahoo Finance API
"""

from __future__ import annotations

import pandas as pd
import requests
from datetime import datetime as dt


class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API via direct HTTP calls (bypasses yfinance crumb auth).

    Attributes
    ----------
        start_date : str
        end_date : str
        ticker_list : list
    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date   = end_date
        self.ticker_list = ticker_list

    def fetch_data(self, proxy=None) -> pd.DataFrame:
        """Fetches adjusted close OHLCV data from Yahoo Finance v8 chart API.

        yfinance >= 0.2.x has a broken crumb/auth mechanism. This method calls
        the raw endpoint directly, which works without session cookies.
        """
        start_ts = int(dt.strptime(self.start_date, "%Y-%m-%d").timestamp())
        end_ts   = int(dt.strptime(self.end_date,   "%Y-%m-%d").timestamp())
        headers  = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

        data_df = pd.DataFrame()
        num_failures = 0

        for tic in self.ticker_list:
            url    = f"https://query1.finance.yahoo.com/v8/finance/chart/{tic}"
            params = {"period1": start_ts, "period2": end_ts, "interval": "1d", "events": "history"}
            try:
                resp   = requests.get(url, params=params, headers=headers, timeout=15)
                result = resp.json()["chart"]["result"]
                if not result or not result[0].get("timestamp"):
                    print(f"No data for {tic}")
                    num_failures += 1
                    continue

                timestamps = result[0]["timestamp"]
                indicators = result[0]["indicators"]
                adj  = indicators.get("adjclose", [{}])[0].get("adjclose")
                raw  = indicators["quote"][0]
                dates = [dt.fromtimestamp(ts).strftime("%Y-%m-%d") for ts in timestamps]

                temp_df = pd.DataFrame({
                    "date":   dates,
                    "open":   raw.get("open"),
                    "high":   raw.get("high"),
                    "low":    raw.get("low"),
                    "close":  adj if adj else raw.get("close"),
                    "volume": raw.get("volume"),
                    "tic":    tic,
                })
                data_df = pd.concat([data_df, temp_df], axis=0)

            except Exception as e:
                print(f"Failed to get ticker '{tic}': {e}")
                num_failures += 1

        if num_failures == len(self.ticker_list):
            raise ValueError("no data is fetched.")

        data_df["date"] = pd.to_datetime(data_df["date"])
        data_df["day"]  = data_df["date"].dt.dayofweek
        data_df["date"] = data_df["date"].dt.strftime("%Y-%m-%d")
        data_df = data_df.dropna().reset_index(drop=True)
        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        return data_df

    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df  = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df


# ===== CELL 13 =====
import datetime
import numpy as np
import pandas as pd
from multiprocessing.sharedctypes import Value

import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf

def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    # _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    _data = pd.read_csv(file_name)
    return _data


def data_split(df, start, end, target_date_col="date"):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data


def convert_to_datetime(time):
    time_fmt = "%Y-%m-%dT%H:%M:%S"
    if isinstance(time, str):
        return datetime.datetime.strptime(time, time_fmt)

# ===== CELL 15 =====
import copy
import datetime
import os
from datetime import date
from datetime import timedelta
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd

# ===== CELL 20 =====
from hyperopt import fmin, tpe, hp, Trials, space_eval


# ===== CELL 26 =====
Nifty_ticker = ['RELIANCE.NS', 'ASIANPAINT.NS', 'BAJFINANCE.NS', 'HDFCBANK.NS', 'SBIN.NS']
sensex_ticker = ["ASIANPAINT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BHARTIARTL.NS", "HCLTECH.NS", "HDFCBANK.NS",
                 "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
                 "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBIN.NS", "SUNPHARMA.NS",
                 "EICHERMOT.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS"]


# BIST Turkey
bist100_top30_tickers = ['AEFES.IS', 'AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'CCOLA.IS',
       'DOHOL.IS', 'EKGYO.IS', 'ENKAI.IS', 'EREGL.IS', 'FROTO.IS', 'GARAN.IS',
       'GOLTS.IS', 'HALKB.IS', 'ISCTR.IS', 'KCHOL.IS', 'KOZAL.IS', 'KRDMD.IS',
       'PETKM.IS', 'SAHOL.IS', 'SISE.IS', 'TAVHL.IS', 'TCELL.IS', 'THYAO.IS',
       'TKFEN.IS', 'TOASO.IS', 'TTKOM.IS', 'TUPRS.IS', 'ULKER.IS', 'VAKBN.IS',
       'VESTL.IS', 'YKBNK.IS']

# Spain IBEX top 30
ibex35_tickers = ['ACS.MC', 'ACX.MC', 'AMS.MC', 'ANA.MC', 'BBVA.MC', 'BKT.MC', 'CABK.MC',
       'COL.MC', 'ELE.MC', 'ENG.MC', 'FDR.MC', 'FER.MC', 'GRF.MC', 'IBE.MC',
       'IDR.MC', 'ITX.MC', 'MAP.MC', 'MEL.MC', 'MTS.MC', 'NTGY.MC', 'RED.MC',
       'REP.MC', 'ROVI.MC', 'SAB.MC', 'SAN.MC', 'SCYR.MC', 'SLR.MC', 'TEF.MC']

# Tickers for the top 30 stocks on B3 (Brasil Bolsa Balcão)

brazil_tickers = ['ABEV3.SA', 'BBAS3.SA', 'BPAN4.SA', 'BRFS3.SA', 'BRKM5.SA', 'CSNA3.SA',
       'CYRE3.SA', 'ECOR3.SA', 'EGIE3.SA', 'ELET3.SA', 'ELET6.SA', 'EMBR3.SA',
       'EQTL3.SA', 'GGBR4.SA', 'ITUB4.SA', 'JBSS3.SA', 'LREN3.SA',
       'MRFG3.SA', 'PETR3.SA', 'PETR4.SA', 'RADL3.SA', 'RENT3.SA', 'SBSP3.SA',
       'SUZB3.SA', 'UGPA3.SA', 'USIM5.SA', 'VALE3.SA', 'WEGE3.SA', 'YDUQ3.SA']


# Final Tickers Hang Seng (Hong Kong)
hang_seng_symbols = ['0002.HK', '0003.HK', '0012.HK', '0017.HK', '0027.HK', '0101.HK',
       '0241.HK', '0267.HK', '0669.HK', '0762.HK', '0836.HK', '0883.HK',
       '0906.HK', '0939.HK', '0992.HK', '1038.HK', '1044.HK', '1093.HK',
       '1109.HK', '1398.HK', '2020.HK', '2319.HK', '2331.HK', '2382.HK',
       '2628.HK', '2688.HK', '3323.HK', '3328.HK', '3983.HK', '3988.HK']

# Tiwan TWSE Market
twse_top30 = ['1216.TW', '1301.TW', '1303.TW', '1519.TW', '1537.TW', '2308.TW',
       '2317.TW', '2330.TW', '2363.TW', '2368.TW', '2382.TW', '2412.TW',
       '2454.TW', '2474.TW', '2504.TW', '2603.TW', '2838.TW', '2880.TW',
       '2881.TW', '2882.TW', '2884.TW', '2886.TW', '2891.TW', '2892.TW',
       '3008.TW', '3045.TW', '3653.TW', '4904.TW', '5880.TW', '6505.TW']
# UK FTSE top 30 working Stock
FTSE_top30 = ['ABF.L', 'ADM.L', 'AHT.L', 'AV.L', 'BA.L', 'BEZ.L', 'CCL.L', 'CNA.L',
       'DPLM.L', 'ENT.L', 'FRAS.L', 'HSBA.L', 'HWDN.L', 'III.L',
       'IMI.L', 'INF.L', 'MKS.L', 'MRO.L', 'NXT.L', 'PSON.L', 'REL.L', 'RR.L',
       'SBRY.L', 'SKG.L', 'SMDS.L', 'SMIN.L', 'SMT.L', 'SPX.L', 'SSE.L']
# Japanies Nikkei Top 30
nikkei_top30_symbols = ['2914.T', '3382.T', '3407.T', '3861.T', '4063.T', '4502.T', '4689.T',
       '4755.T', '5802.T', '6301.T', '6471.T', '6501.T', '6594.T', '6701.T',
       '6758.T', '6920.T', '7011.T', '7203.T', '7267.T', '7735.T', '7974.T',
       '8031.T', '8035.T', '8058.T', '8306.T', '8316.T', '9020.T', '9022.T',
       '9983.T', '9984.T']
# German DAX top 30
dax_30 = ['ADS.DE', 'AIR.DE', 'ALV.DE', 'BAS.DE', 'BEI.DE', 'BMW.DE', 'BNR.DE',
       'BOSS.DE', 'CBK.DE', 'CON.DE', 'DB1.DE', 'DBK.DE', 'DTE.DE', 'DWNI.DE',
       'EOAN.DE', 'EVT.DE', 'FME.DE', 'FNTN.DE', 'FRE.DE', 'HEI.DE', 'HNR1.DE',
       'LIN.DE', 'MRK.DE', 'MTX.DE', 'MUV2.DE', 'SAP.DE', 'SIE.DE', 'SY1.DE',
       'TL0.DE', 'VOW3.DE']
# USA Dow 30
Dow_30 = ['AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'GS',
       'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK',
       'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT']

indices= [sensex_ticker, Dow_30, dax_30, nikkei_top30_symbols, FTSE_top30, twse_top30, hang_seng_symbols, brazil_tickers, ibex35_tickers, bist100_top30_tickers ]


# Load from cached CSV and filter to last 3 years
# (1 yr for 252-day covariance lookback + 2 yr training window)
df = pd.read_csv('artifacts/data.csv')
df = df[df['date'] >= '2022-01-01'].reset_index(drop=True)
print(f"Loaded {df.shape[0]} rows | {df.date.min()} -> {df.date.max()} | {df.tic.nunique()} tickers")

# ===== CELL 28 =====
from stockstats import StockDataFrame as Sdf

def add_tech(data, INDICATORS):
  df = data.copy()
  df = df.sort_values(by=["tic", "date"])
  stock = Sdf.retype(df.copy())
  unique_ticker = stock.tic.unique()

  for indicator in INDICATORS:
      indicator_df = pd.DataFrame()
      for i in range(len(unique_ticker)):
          try:
              temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
              temp_indicator = pd.DataFrame(temp_indicator)
              temp_indicator["tic"] = unique_ticker[i]
              temp_indicator["date"] = df[df.tic == unique_ticker[i]][
                  "date"
              ].to_list()
              # indicator_df = indicator_df.append(
              #     temp_indicator, ignore_index=True
              # )
              indicator_df = pd.concat(
                  [indicator_df, temp_indicator], axis=0, ignore_index=True
              )
          except Exception as e:
              print(e)
      df = df.merge(
          indicator_df[["tic", "date", indicator]], on=["tic", "date"], how="left"
      )

  df = df.sort_values(by=["date", "tic"])

  return df

# ===== CELL 29 =====
INDICATORS = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma']
df = add_tech(df, INDICATORS)
df = df.ffill().bfill()

# ===== CELL 30 =====
# add covariance matrix as states
df=df.sort_values(['date','tic'],ignore_index=True)
df.index = df.date.factorize()[0]

cov_list = []
return_list = []

# look back is one year
lookback=252
for i in range(lookback,len(df.index.unique())):
  data_lookback = df.loc[i-lookback:i,:]
  price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
  return_lookback = price_lookback.pct_change().dropna()
  return_list.append(return_lookback)

  covs = return_lookback.cov().values
  cov_list.append(covs)


df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
df = df.merge(df_cov, on='date')
df = df.sort_values(['date','tic']).reset_index(drop=True)



# ===== CELL 34 =====
print(df.shape)

hist_vol=[]
for i in range(len(df['return_list'])):
  returns = df['return_list'].values[i].std()
  hist_vol.append(returns)
print(len(hist_vol))



# ===== CELL 35 =====
hist_vol = np.array(hist_vol)
hist_vol = pd.DataFrame(hist_vol, df['date'])
# Optionally save locally:
# df.to_csv('artifacts/sensex_data.csv')
# hist_vol.to_csv('artifacts/sensex_hist_vol.csv')


# ===== CELL 39 =====
import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv

# ===== CELL 41 =====
# ── Last 2 Years: Data Splits ─────────────────────────────────────────────────
# Raw data loaded from 2022-01-01; 252-day covariance lookback burns through 2022.
# Usable training data starts 2023-01-06.

TRAIN_START_DATE = '2023-01-01'
TRAIN_END_DATE   = '2023-12-31'   # 1 yr  (hyperparameter search env)

Val_START_DATE   = '2024-01-01'
VAL_END_DATE     = '2024-06-30'   # 6 mo  (validation)

FULL_TRAIN_START = '2023-01-01'
FULL_TRAIN_END   = '2024-12-31'   # 2 yrs (final agent training)

TRADE_START_DATE = '2025-01-01'
TRADE_END_DATE   = '2025-04-30'   # test period

train          = data_split(df, TRAIN_START_DATE, TRAIN_END_DATE)
hist_vol_train = hist_vol[TRAIN_START_DATE : TRAIN_END_DATE]

val            = data_split(df, Val_START_DATE, VAL_END_DATE)
hist_vol_val   = hist_vol[Val_START_DATE : VAL_END_DATE]

full_train          = data_split(df, FULL_TRAIN_START, FULL_TRAIN_END)
hist_vol_full_train = hist_vol[FULL_TRAIN_START : FULL_TRAIN_END]

trade          = data_split(df, TRADE_START_DATE, TRADE_END_DATE)
hist_vol_trade = hist_vol[TRADE_START_DATE : TRADE_END_DATE]

print(f"Train rows     : {train.shape[0]}  ({TRAIN_START_DATE} -> {TRAIN_END_DATE})")
print(f"Val rows       : {val.shape[0]}  ({Val_START_DATE} -> {VAL_END_DATE})")
print(f"Full train rows: {full_train.shape[0]}  ({FULL_TRAIN_START} -> {FULL_TRAIN_END})")
print(f"Trade rows     : {trade.shape[0]}  ({TRADE_START_DATE} -> {TRADE_END_DATE})")


# ===== CELL 43 =====
class StockPortfolioEnv(gym.Env):
    """A single stock trading environment for OpenAI gym

    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            a threshold to control risk aversion
        day: int
            an increment number to control date

    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step


    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                df,
                stock_dim,
                hmax,
                initial_amount,
                transaction_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                turbulence_threshold=None,
                lookback=252,
                day = 0, hist_vol= None):

        self.day = day
        self.lookback=lookback
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct =transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.hist_vol=hist_vol
        self.DSR_A = 0.0
        self.DSR_B = 0.0

         # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low = 0, high = 1,shape = (self.action_space,))

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space+1 + len(self.tech_indicator_list), self.state_space))



        self.data = self.df.loc[self.day,:]
        self.covs = self.data['cov_list'].values[0]




        self.state = np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
        # print(" state  :: " , self.day ,self.state.shape, self.state)
        # print(" hist_ vol  :: " , self.day , type(self.hist_vol), self.hist_vol)

        hist_volll = self.hist_vol.values[self.day,:]
        # Concatenate along axis=0

        self.state = np.concatenate([self.state, hist_volll.reshape(1,-1) ], axis=0)



        # print("states - " , self.state.shape)

        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        self.date_memory=[self.data.date.unique()[0]]



    def step(self, actions):
      print(f" the len of the df is  {len(self.df.index.unique())}  and the current day is :  {self.day } and  if  terminal is  : { self.day >= len(self.df.index.unique()) - 1 }")
      self.terminal = self.day >= len(self.df.index.unique()) - 1

      if self.terminal:
          # print("=================================")
          # print("begin_total_asset:{}".format(self.asset_memory[0]))
          # print("end_total_asset:{}".format(self.portfolio_value))
          # return self.state, self.reward, self.terminal, {}


          df = pd.DataFrame(self.portfolio_return_memory)
          df.columns = ['daily_return']
          # plt.plot(df.daily_return.cumsum(),'r')
          # plt.savefig('results/cumulative_reward.png')
          # plt.close()

          # plt.plot(self.portfolio_return_memory,'r')
          # plt.savefig('results/rewards.png')
          # plt.close()

          print("=================================")
          print("begin_total_asset:{}".format(self.asset_memory[0]))
          print("end_total_asset:{}".format(self.portfolio_value))

          df_daily_return = pd.DataFrame(self.portfolio_return_memory)
          df_daily_return.columns = ['daily_return']
          if df_daily_return['daily_return'].std() !=0:
            sharpe = (252**0.5)*df_daily_return['daily_return'].mean()/ \
                    df_daily_return['daily_return'].std()
            print("Sharpe: ",sharpe)
          print("=================================")


          return self.state, self.reward, self.terminal,{}
      else:
          last_day_memory = self.data
          weights = self.softmax_normalization(actions)  # Ensure valid portfolio weights
          self.actions_memory.append(weights)

          # Load next state
          self.day = self.day+ 1
          self.data = self.df.loc[self.day, :]
          self.covs = self.data['cov_list'].values[0]
          self.state = np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
          hist_voll= self.hist_vol.values[self.day,:]
          self.state = np.concatenate([self.state, hist_voll.reshape(1,-1) ], axis=0)

          # Portfolio Value Update
          portfolio_return = sum(((self.data.close.values / last_day_memory.close.values) - 1) * weights)
          new_portfolio_value = self.portfolio_value * (1 + portfolio_return)

          # Calculate Transaction Fee
          phi = 0.0025  # 0.25% transaction cost
          # Reshape portfolio_value to match dimensions of other arrays
          portfolio_value_reshaped = np.repeat(self.portfolio_value, len(weights))
          transaction_fee = phi * sum(
              abs(weights * new_portfolio_value * last_day_memory.close.values / self.data.close.values
                  - self.actions_memory[-2] * portfolio_value_reshaped)  # Use portfolio_value_reshaped
          )

          # Reward Calculation
          self.reward = (new_portfolio_value - self.portfolio_value) - transaction_fee  # r_t = u_t - u_t-1 - fee_t

          # Update portfolio value
          self.portfolio_value = new_portfolio_value

          # Save to memory
          self.portfolio_return_memory.append(portfolio_return)
          self.asset_memory.append(new_portfolio_value)
          self.date_memory.append(self.data.date.unique()[0])

          return self.state, self.reward, self.terminal, {}
    ##############################################




    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0

        # returns = self.df['return_list'].values[0]
        # hist_vol = returns.rolling(window=30).std()
        # hist_vol.fillna(0, inplace=True)
        # hist_vol = hist_vol.iloc[self.day,:]


        self.data = self.df.loc[self.day,:]
        # load states
        self.covs = self.data['cov_list'].values[0]
        self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
        # print(self.hist_vol)
        # self.hist_vol= self.hist_vol[self.day,]
        # Concatenate along axis=0

        hist_voll= self.hist_vol.values[self.day,:]
        self.state = np.concatenate([self.state, hist_voll.reshape(1,-1)], axis=0)
        # Concatenate along axis=0





        # print(" reset -- ev  --state -", self.state.shape)
        # print(" reset -- ev -- state - ", self.state)
        # print(" reset -- ev-- cov - ", self.state[:30, :].shape)
        # print(" reset -- ev-- his vol- ", self.state[:-1, :].shape)
        # print(" reset -- ev-- his vol- ", self.state[-1:, :])
        self.portfolio_value = self.initial_amount
        #self.cost = 0
        #self.trades = 0
        self.DSR_A = 0.0
        self.DSR_B = 0.0
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        self.date_memory=[self.data.date.unique()[0]]
        return self.state

    def render(self, mode='human'):
        return self.state

    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator/denominator
        return softmax_output


    def apply_dirichlet_noise(self, actions, alpha=0.1):
      """
      Apply Dirichlet noise to actions to encourage exploration.

      Args:
      - actions (np.array): Original action values from the RL model.
      - alpha (float): Dirichlet concentration parameter. Lower values = more noise.

      Returns:
      - np.array: Modified action values with noise, ensuring sum = 1.
      """
      noise = np.random.dirichlet([alpha] * len(actions))  # Sample from Dirichlet distribution
      noisy_actions = 0.75 * actions + 0.25 * noise  # Blend original actions with noise
      return noisy_actions / noisy_actions.sum()  # Normalize to ensure sum = 1




    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        #print(len(date_list))
        #print(len(asset_list))
        df_account_value = pd.DataFrame({'date':date_list,'daily_return':portfolio_return})
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def calculate_DSR(self, R):
      eta = 0.004
      delta_A = R - self.DSR_A
      delta_B = R**2 - self.DSR_B
      Dt = (self.DSR_B*delta_A - 0.5*self.DSR_A*delta_B) / ((self.DSR_B-self.DSR_A**2)**(3/2) + 1e-6)
      self.DSR_A = self.DSR_A + eta*delta_A
      self.DSR_B = self.DSR_B + eta*delta_B
      return(Dt)

# ===== CELL 44 =====
stock_dimension = len(train.tic.unique())

state_space = stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

# ===== CELL 45 =====
# print(INDICATORS)
TURBULENCE_THRESHOLD= 0.0020

env_kwargs_train = {
    "hmax": 100,
    "initial_amount": 1000000,
    "transaction_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
    "hist_vol":hist_vol_train,
    'turbulence_threshold': TURBULENCE_THRESHOLD

}
# print(hist_vol_val,"  ddddd ")
env_kwargs_val = {
    "hmax": 100,
    "initial_amount": 1000000,
    "transaction_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
    "hist_vol":hist_vol_val,
    "turbulence_threshold": TURBULENCE_THRESHOLD
}

env_kwargs_full = {
    "hmax": 100,
    "initial_amount": 1000000,
    "transaction_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
    "hist_vol":hist_vol_full_train,
    "turbulence_threshold": TURBULENCE_THRESHOLD
}

env_kwargs_trade = {
    "hmax": 100,
    "initial_amount": 1000000,
    "transaction_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
    "hist_vol":hist_vol_trade,
    "turbulence_threshold": TURBULENCE_THRESHOLD
}



# ===== CELL 46 =====
e_train_gym = StockPortfolioEnv(df = train, **env_kwargs_train)
env_train, _ = e_train_gym.get_sb_env()

e_val_gym = StockPortfolioEnv(df = val, **env_kwargs_val)
env_val, _ = e_val_gym.get_sb_env()

e_train_full_gym = StockPortfolioEnv(df = full_train, **env_kwargs_full)
env_full_train, _ = e_train_full_gym.get_sb_env()

e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs_trade)
env_trade, _ = e_trade_gym.get_sb_env()
print("done")

# ===== CELL 48 =====
import random
from collections import deque

class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)

        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_state_batch = np.array(next_state_batch)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)

# ===== CELL 50 =====

import numpy as np


def Noise(action, action_space, kappa=10):
    """
    Apply Dirichlet noise for exploration in DDPG according to the paper.

    Args:
    - action (torch.Tensor): Original action values from the actor network.
    - action_space (gym.spaces.Box): Action space defining valid ranges.
    - kappa (float): Controls exploration variance. Higher kappa = less noise.

    Returns:
    - np.array: Modified action values with Dirichlet noise, ensuring sum = 1.
    """

    try:
        # Ensure actions are non-negative before applying Dirichlet noise
        action = torch.clamp(action, min=0.0)

        # Convert actions to numpy array for Dirichlet sampling
        action_np = action.detach().cpu().numpy()

        # Compute shape parameter: υ = κ * a
        upsilon = kappa * action_np

        # Ensure upsilon is positive and correctly shaped
        upsilon = np.maximum(upsilon, 1e-6)  # Prevent zero or negative values
        upsilon = upsilon.flatten()  # Ensure it's a 1D array

        # Debugging: Check upsilon values
        if np.any(upsilon <= 0):
            raise ValueError(f"Dirichlet parameters must be positive. Found: {upsilon}")

        # Sample ϵ from Dirichlet distribution
        epsilon = np.random.dirichlet(upsilon)

        # Compute final action: a' = a + sg(ϵ - a)
        noisy_action = action_np + (epsilon - action_np)

        # Apply StopGradient (detach the noise term)
        noisy_action = action_np + torch.tensor(noisy_action - action_np, requires_grad=False).numpy()

        # Clip extreme values to prevent instability
        noisy_action = np.clip(noisy_action, 0.0, 1.0)

        # Ensure sum = 1 for valid portfolio allocation
        noisy_action = noisy_action / noisy_action.sum()

        return noisy_action

    except ValueError as ve:
        print(f"ValueError in Dirichlet noise function: {ve}")
    except Exception as e:
        print(f"Unexpected error in Dirichlet noise function: {e}")

    # Return the original action if an error occurs
    return action.detach().cpu().numpy()




# ===== CELL 51 =====
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers, act_fn, dr):
        super(Actor, self).__init__()

        layers = []

        if act_fn == 'relu': activation_fn = nn.ReLU()
        if act_fn == 'tanh': activation_fn = nn.Tanh()
        if act_fn == 'sigmoid': activation_fn = nn.Sigmoid()
        # print("Params Dictionary:", self.params)
        hidden_dim = int(hidden_dim)
        num_layers = int(num_layers)
        action_dim = int(action_dim)
        state_dim = int(state_dim)

        # print("state_dim:", state_dim)
        # print("action_dim:", action_dim)
        # print("hidden_dim:", hidden_dim)
        # print("num_layers:", num_layers)
        # print("act_fn:", act_fn)
        # print("dr:", dr)
        # print(f"state_dim: {state_dim}, type: {type(state_dim)}")
        # print(f"action_dim: {action_dim}, type: {type(action_dim)}")
        # print(f"hidden_dim: {hidden_dim}, type: {type(hidden_dim)}")

        # Add input layer

        layers.append(nn.Flatten())
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(activation_fn)
        layers.append(nn.Dropout(p=dr))

        # Add hidden layers
        for _ in range(num_layers - 2):  # -2 because we already added the input and output layers
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn)
            layers.append(nn.Dropout(p=dr))

        # Add output layer
        layers.append(nn.Linear(hidden_dim, action_dim))
        # layers.append(nn.Dropout(p=dr))

        # Create the sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, state):

        x = self.model(state)
        x = torch.tanh(x)
        # print(" actor  Network forward (((((((((((((((((((((((((((((((((((((())))))))))))))))))))))))))))))))))))))")
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers, act_fn, dr):
        super(Critic, self).__init__()

        layers = []

        if act_fn == 'relu': activation_fn = nn.ReLU()
        if act_fn == 'tanh': activation_fn = nn.Tanh()
        if act_fn == 'sigmoid': activation_fn = nn.Sigmoid()
        hidden_dim = int(hidden_dim)
        num_layers = int(num_layers)
        action_dim = int(action_dim)
        state_dim = int(state_dim)

        # print("state_dim:", state_dim)
        # print("action_dim:", action_dim)
        # print("hidden_dim:", hidden_dim)
        # print("num_layers:", num_layers)
        # print("act_fn:", act_fn)
        # print("dr:", dr)
        # print(f"state_dim: {state_dim}, type: {type(state_dim)}")
        # print(f"action_dim: {action_dim}, type: {type(action_dim)}")
        # print(f"hidden_dim: {hidden_dim}, type: {type(hidden_dim)}")


        # Add input layer
        layers.append(nn.Linear(state_dim + action_dim, hidden_dim))
        layers.append(activation_fn)
        layers.append(nn.Dropout(p=dr))

        # Add hidden layers
        for _ in range(num_layers - 2):  # -2 because we already added the input and output layers
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn)
            layers.append(nn.Dropout(p=dr))

        # Add output layer
        # layers.append(nn.Dropout(p=dr))
        layers.append(nn.Linear(hidden_dim, 1))

        # Create the sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, state, action):
        """
        Forward pass of the Critic network.

        Args:
        - state (torch.Tensor): State tensor.
        - action (torch.Tensor): Action tensor.

        Returns:
        - Q-value estimation.
        """

        # 🔍 Print debug info
        # print("Critic Network forward (((((((((((((((((((((((((((((((((((((())))))))))))))))))))))))))))))))))))))")
        # print(f"State shape before reshape: {state.shape}, Action shape before reshape: {action.shape}")

        # 🔄 Flatten state if it has more than 2 dimensions (CNN case)
        if state.dim() > 2:
            state = state.view(state.shape[0], -1)  # Convert to (batch_size, features)

        # 🔄 Ensure action is 2D
        if action.dim() > 2:
            action = action.view(action.shape[0], -1)  # Convert to (batch_size, action_dim)

        # 🔍 Print final shapes
        # print(f"State shape after reshape: {state.shape}, Action shape after reshape: {action.shape}")

        # ✅ Now both state and action are 2D → Safe to concatenate
        x = torch.cat([state, action], dim=1)

        # Forward pass through Critic layers
        x = self.model(x)

        return x





class CostNetwork(nn.Module):
    """
    Neural network for estimating portfolio risk (cost).
    """
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(CostNetwork, self).__init__()

        state_dim=int(state_dim)
        action_dim=int(action_dim)
        hidden_dim=int(hidden_dim)
        # print("state_dim:", state_dim)
        # print("action_dim:", action_dim)
        # print("hidden_dim:", hidden_dim)

        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Outputs cost estimate
        )

    def forward(self, state, action):
        """
        Forward pass for the cost network.

        Computes:
        c_wv(s, a) = E[VaR(s, a)]  (Eq. 19 in the paper)

        Args:
        - state (torch.Tensor): State tensor with shape [batch_size, *]
        - action (torch.Tensor): Action tensor with shape [batch_size, action_dim]

        Returns:
        - Cost estimation (torch.Tensor)
        """

        # print(" cost network forward ((((((((((((((((((((((((((((((((((((((()))))))))))))))))))))))))))))))))))))))")
        # 🔍 Print debug info to check tensor shapes
        # print("state :: ", type(state) , state.shape)
        # print("action :: ", type(action) , action.shape)
        # 🔄 Flatten state if it has more than 2 dimensions
        if state.dim() > 2:
            state = state.view(state.shape[0], -1)  # Reshape to [batch_size, flattened_features]

        # 🔄 Ensure action is 2D
        if action.dim() > 2:
            action = action.view(action.shape[0], -1)  # Reshape to [batch_size, action_dim]

        # 🔍 Print final shapes
        # print(f"State shape after reshape: {state.shape}, Action shape after reshape: {action.shape}")

        # ✅ Now both state and action are 2D → Safe to concatenate
        x = torch.cat([state, action], dim=1)
        # Forward pass through the Cost network
        return self.model(x)





# ===== CELL 52 =====
#device = 'cpu'
# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== CELL 53 =====
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import norm  # For z-score

class DDPGagent:
    def __init__(self, env, params, max_memory_size=50000):
        """
        Initialize the DDPG agent with:
        - Actor-Critic Networks
        - Cost Network for risk constraints
        - Target Networks for stability
        - Lagrange multiplier for enforcing constraints
        """

        # print(params)
        print(" DDPG AGEnt Class- ++++++++++++++++++++++++++++++++++++++++++")

        # 1️⃣ Define State & Action Space Dimensions
        self.data = env.envs[0].df
        curr_state= env.envs[0].state
        # print("states_ ddpg init ::", curr_state.shape)
        actions = env.action_space.shape[0]

        # print("actions ::", actions)

        self.num_states = env.observation_space.shape[0] * env.observation_space.shape[1]
        self.num_actions = env.action_space.shape[0]
        self.gamma = params['gamma']  # Discount factor (γ)
        self.tau = params['tau']  # Soft update factor (τ)
        self.batch_size = int(params['batch_size'])
        self.env = env
        self.eta = params['eta']


        # 2️⃣ Initialize Networks
        self.actor = Actor(self.num_states, self.num_actions, params['Ahidden_dim'],
                           params['Anum_layers'], params['Aact_fn'], params['Adr']).to(device)
        self.actor_target = Actor(self.num_states, self.num_actions, params['Ahidden_dim'],
                                  params['Anum_layers'], params['Aact_fn'], params['Adr']).to(device)

        self.critic = Critic(self.num_states, self.num_actions, params['Chidden_dim'],
                             params['Cnum_layers'], params['Cact_fn'], params['Cdr']).to(device)
        self.critic_target = Critic(self.num_states, self.num_actions, params['Chidden_dim'],
                                    params['Cnum_layers'], params['Cact_fn'], params['Cdr']).to(device)

        # 3️⃣ Initialize Cost Network for Constrained Reinforcement Learning
        self.cost_network = CostNetwork(self.num_states, self.num_actions, params['Chidden_dim']).to(device)
        self.cost_target = CostNetwork(self.num_states, self.num_actions, params['Chidden_dim']).to(device)

        # Copy weights to target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.cost_target.parameters(), self.cost_network.parameters()):
            target_param.data.copy_(param.data)

        # 4️⃣ Training Setup
        self.memory = Memory(max_memory_size)
        self.critic_criterion = nn.MSELoss()
        self.cost_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=params['alr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=params['clr'])
        self.cost_optimizer = optim.Adam(self.cost_network.parameters(), lr=params['clr'])

        # 5️⃣ Initialize Lagrange Multiplier for Constraint Enforcement
        self.lambda_ = 0.01
        self.rho = 0.01  # Step size for updating lambda
        self.violations= 0
        self.zeta= env.envs[0].turbulence_threshold


    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).to(device)
        action = self.actor.forward(state_tensor).detach().cpu()

        #action = action.detach().numpy()
        return action



    def VaR(self, states, actions, confidence_level=0.95):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        actions = actions.to(device)
        states = states.to(device)  # assume actions is already on the correct device

        batch_size = states.shape[0]  # ✅ Do NOT use `.to(device)` here
        num_assets = 30

        states = states.squeeze(1).to(device)  # [batch_size, 38, 30]
        states_n = states  # already squeezed

        cov_matrix = states[:, :num_assets, :].to(device)  # [batch_size, 30, 30]
        hist_volatility = states_n[:, -1, :].to(device)  # [batch_size, 30]

        z_score = torch.tensor(1.645, device=device)  # ✅ place tensor on the same device
        individual_VaR = z_score * hist_volatility  # [batch_size, 30]

        VaR_portfolio = torch.zeros(batch_size, device=device)  # ✅ directly initialize on device

        for i in range(num_assets):
            for j in range(num_assets):
                VaR_portfolio = VaR_portfolio + (
                    actions[:, i] * individual_VaR[:, i] *
                    actions[:, j] * individual_VaR[:, j] * cov_matrix[:, i, j]
                )

        return VaR_portfolio







    def compute_cost_target(self, states, actions, next_states, dones):
        """
        Compute the target cost using the Bellman equation.

        Equation (20):
        c_{w_v}(s, a) = VaR(s, a) + \eta (1 - d) c'_{w_v'}(s', a')
        """
        next_actions = self.actor_target.forward(next_states)  # π'(s')
        next_cost = self.cost_target.forward(next_states, next_actions.detach())  # c'_wv'(s', a')
        cost_target = self.VaR(next_states, next_actions) + self.eta * next_cost
        return cost_target

    def update(self):
        """
        Perform one update step for the Actor, Critic, and Cost networks.
        """

        # 1️⃣ Sample a batch from the Replay Buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)


        # print(" update states : ", type(states),  states.shape,  " action ", action.shape, type(action) )

        # Remove the singleton dimension at dim=1
        # states_n = states.squeeze(1)  # shape: (224, 38, 30)

        # # Now slicing makes sense
        # cov_mat = states_n[:, :30, :]                  # Shape: (224, 30, 30)
        # histrical_volatility = states_n[:, -1, :]


        # print("ddpg update - states_n ::", states_n.shape)
        # print("ddpg update - cov_mat_n ::", cov_mat.shape)
        # print("ddpg update - histrical_volatility_n ::", histrical_volatility.shape)

        # print("ddpg update - states_n ::", states_n)
        # print("ddpg update - cov_mat_n ::", cov_mat)
        # print("ddpg update - histrical_volatility_n ::", histrical_volatility)


        # next_states_n = next_states.squeeze(1)  # Shape: [batch_size, 38, 30]
        # next_cov_mat = next_states_n[:, :30, :]  # Shape: [batch_size, 30, 30]
        # next_hist_volatility = next_states_n[:, -1, :]
        # print("ddpg update - next_states_n ::", next_states_n.shape)
        # print("ddpg update - next_cov_mat ::", next_cov_mat.shape)
        # print("ddpg update - histor vol :: " , next_hist_volatility.shape)
        # print("ddpg update - next_states_n ::", next_states_n)
        # print("ddpg update - next_cov_mat ::", next_cov_mat)
        # print("ddpg update - histor vol :: " , next_hist_volatility)







        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # 4️⃣ Compute Target Q-Value using Bellman Equation (Eq. 5)
        # Q(s, a) = r + γQ'(s', π'(s'))
        Q_target = rewards + self.gamma  * self.critic_target.forward(next_states, self.actor_target.forward(next_states).detach())

        # 6️⃣ Compute Critic Loss (Eq. 6)
        # L = 1/N \sum (Q(s, a) - Q_target)^2
        # print(" critic loss calculation -start")
        critic_loss = self.critic_criterion(self.critic.forward(states, actions), Q_target.detach())

        # print("critic loss calculation end ")
        # 8️⃣ Compute Cost Network Loss (Eq. 21)
        # L_C = 1/N \sum (c_{w_v}(s, a) - VaR(s, a) - η (1 - d) c'_{w_v'}(s', a'))^2
        # print("cost loss calculation started ")
        cost_pred = self.cost_network.forward(states, actions)
        cost_target = self.compute_cost_target(states, actions, next_states, dones).detach()
        # print(" cost_ target :: " , cost_target)

        cost_loss = self.cost_criterion(cost_pred, cost_target)

        # print("cost loss calculation end ")
        # 🔟 Compute Actor Loss using Lagrangian method (Eq. 13)
        # L(w_π, λ) = -J_{w_π} + \sum \lambda_j C_{w_π, j} + \frac{\rho}{2} \sum (C_{w_π, j})^2

        # print("actor loss calculation started ")
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        constraint_penalty =  cost_target

        # print(" constraint penalty before :: ", constraint_penalty)
        # print(" constraint_penalty :::: " , constraint_penalty.shape, type(constraint_penalty))

        violations_count = (constraint_penalty > self.zeta).sum().item()  # Count how many elements violate the constraint
        # print(" violations ::: " , violations_count)
        # Update the number of violations
        self.violations  =  self.violations + violations_count


        constraint_penalty = torch.where(
            constraint_penalty <= self.zeta,
            torch.tensor(0.0, device=constraint_penalty.device, dtype=constraint_penalty.dtype),
            constraint_penalty - self.zeta
        )
        # print(" constraint penalty after :: ", constraint_penalty)

        quadratic_penalty = (self.rho / 2) * (constraint_penalty ** 2).mean().clone()
        constraint_penalty=(self.lambda_ * constraint_penalty).mean().clone()
        actor_loss = policy_loss + constraint_penalty + quadratic_penalty

        self.actor_optimizer.zero_grad()
        # print(" actor_ loss ",  actor_loss.shape)
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.actor_optimizer.step()


        # print("actor update end ")



        # 1️⃣3️⃣ Soft Update of Target Networks (Eq. 14)
        # print("soft update - critic -")
        with torch.no_grad():
          for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
              target_param.data= param.data * self.tau + target_param.data * (1.0 - self.tau)

          for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
              target_param.data = param.data * self.tau + target_param.data * (1.0 - self.tau)

          for target_param, param in zip(self.cost_target.parameters(), self.cost_network.parameters()):
              target_param.data = param.data * self.tau + target_param.data * (1.0 - self.tau)




        # print(" soft updates end")

    def buffer_fill(self, buffer_size):
      state = self.env.reset()

      # print(" buffer fill ------ ")
      # print(" buffer fil state   --- ",  state.shape)
      # print("  buffer fill state  --- ", state)
      # print(" buffer --- fill -- cov mat", state[:, :30, :].shape)

      # print(" buffer fill -----hist vol", state[:, -1, :].shape)
      # print(" buffer fill -----hist vol", state[:, -1, :])

      for _ in range(buffer_size):
        action = self.get_action(state)
        action = Noise(action, self.env.action_space)
        new_state, reward, done, _ = self.env.step(action)
        self.memory.push(state, action, reward, new_state, done)

    def trade(self, val_env, e_val_gym):
      Reward = []
      state = val_env.reset()

      for i in range(len(e_val_gym.df.index.unique())):
        action = self.get_action(state)
        next_obs, reward, done, _ = val_env.step(action.detach().numpy())
        Reward.append(reward)

        if i == (len(e_val_gym.df.index.unique()) - 2):
          account_memory = val_env.env_method(method_name="save_asset_memory")
          actions_memory = val_env.env_method(method_name="save_action_memory")

        if done[0]:
          print("hit end!")
          break
        state = next_obs

      return account_memory, actions_memory, sum(Reward)



# ===== CELL 54 =====
#Calculate the Sharpe ratio
#This is our objective for tuning
def calculate_sharpe(df):
  #df['daily_return'] = df['account_value'].pct_change(1)
  if df['daily_return'].std() !=0:
    sharpe = (252**0.5)*df['daily_return'].mean()/ \
          df['daily_return'].std()
    return sharpe
  else:
    return 0

# ===== CELL 55 =====
space = {
    'Ahidden_dim': hp.quniform('Ahidden_dim', 2, 512, 1),
    'Anum_layers': hp.quniform('Anum_layers', 1, 8, 1),
    'Chidden_dim': hp.quniform('Chidden_dim', 2, 512, 1),
    'Cnum_layers': hp.quniform('Cnum_layers', 1, 8, 1),

    'alr': hp.loguniform('alr', -8, -1),  # Actor learning rate
    'clr': hp.loguniform('clr', -8, -1),  # Critic learning rate
    'gamma': hp.uniform('gamma', 0.9, 0.99),  # Discount factor
    'tau': hp.uniform('tau', 0.08, 0.2),  # Soft target update rate
    'batch_size': hp.quniform('batch_size', 32, 256, 32),  # Mini-batch size

    'Aact_fn': hp.choice('Aact_fn', ['relu', 'tanh', 'sigmoid']),  # Actor activation
    'Adr': hp.uniform('Adr', 0, 0.5),  # Actor dropout
    'Cact_fn': hp.choice('Cact_fn', ['relu', 'tanh', 'sigmoid']),  # Critic activation
    'Cdr': hp.uniform('Cdr', 0, 0.5),  # Critic dropout


    'eta' : hp.uniform('eta', 0.01, 1),  # Exploration noise level
    #  **Newly Added Missing Hyperparameters**:
    # 'rho': hp.uniform('rho', 0.001, 0.1),  # Lagrange multiplier update step size
    # 'lambda_init': hp.uniform('lambda_init', 0.01, 1.0),  # Initial value of λ
    'buffer_size': hp.quniform('buffer_size', 10000, 1000000, 10000),  # Replay buffer size
    'noise_std': hp.uniform('noise_std', 0.01, 0.3),  # Exploration noise level
    'grad_clip': hp.uniform('grad_clip', 0.1, 10.0),  # Gradient clipping threshold
    'warmup_steps': hp.quniform('warmup_steps', 1000, 50000, 1000),  # Steps before training starts
    'reward_scaling': hp.uniform('reward_scaling', 0.1, 10.0)  # Reward scaling factor
}


def objective(params):
    print(params)
    # Convert hyperparameters to integers where necessary
    params['Ahidden_dim'] = int(params['Ahidden_dim'])
    params['Anum_layers'] = int(params['Anum_layers'])
    params['Chidden_dim'] = int(params['Chidden_dim'])
    params['Cnum_layers'] = int(params['Cnum_layers'])
    params['batch_size'] = int(params['batch_size'])
    params['buffer_size'] = int(params['buffer_size'])
    params['warmup_steps'] = int(params['warmup_steps'])

    model = DDPGagent(env_train, params)
    model.buffer_fill(500)
    model.update()

    account_memory, actions_memory, rewardd = model.trade(env_val, e_val_gym)
    print( f" the reward is :::::::    {rewardd}  " )

    sharpe = calculate_sharpe(account_memory[0])
    return -sharpe
    # return -reward[0]

# ===== CELL 56 =====
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals= 10 , trials=Trials()) #max_evals = 500

# ===== CELL 57 =====
best['Aact_fn'] = ['relu', 'tanh', 'sigmoid'][best['Aact_fn']]
best['Cact_fn'] = ['relu', 'tanh', 'sigmoid'][best['Cact_fn']]
best

# ===== CELL 59 =====
agent = DDPGagent(env_full_train, best)

batch_size = agent.batch_size

# ===== CELL 60 =====
import json as _json, os as _os

rewards      = []
avg_rewards  = []
num_episodes = 1   # increase to train longer

torch.autograd.set_detect_anomaly(True)
_os.makedirs("artifacts", exist_ok=True)

for episode in range(num_episodes):
    state          = env_full_train.reset()
    episode_reward = 0
    done           = False

    print(f"Episode: {episode+1}/{num_episodes}")
    while not done:
        action = agent.get_action(state)
        action = Noise(action, env_full_train.action_space)
        new_state, reward, done, info = env_full_train.step(action)
        agent.memory.push(state, action, reward, new_state, done)

        if len(agent.memory) > batch_size:
            agent.update()

        state          = new_state
        episode_reward += reward

        if done:
            break

    # Lagrange multiplier update
    device        = next(agent.cost_network.parameters()).device
    state_tensor  = torch.tensor(np.expand_dims(state, axis=1), dtype=torch.float32, device=device)
    action_tensor = agent.get_action(state).to(device)
    agent.lambda_ = agent.lambda_ + agent.rho * agent.cost_network.forward(
        state_tensor, action_tensor
    ).mean().detach().to(device)
    agent.rho *= 1.008

    rewards.append(float(episode_reward))
    avg_rewards.append(float(np.mean(rewards[-10:])))

    print(f"  Total Reward : {episode_reward:.4f}")
    print(f"  Violations   : {agent.violations}")

    # Save agent weights
    agent.save("artifacts/ddpg_agent")

    # Save rewards so the plot cell works without retraining
    with open("artifacts/training_rewards.json", "w") as _f:
        _json.dump({"rewards": rewards, "avg_rewards": avg_rewards}, _f)

print("Training complete.")
print("Agent saved   -> artifacts/ddpg_agent")
print("Rewards saved -> artifacts/training_rewards.json")

