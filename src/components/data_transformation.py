from stockstats import StockDataFrame as Sdf
import os 
import numpy as np
import sys
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    INDICATORS = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma', 'close_60_sma']

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def add_tech(self, data, INDICATORS):
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
                    raise CustomException(e,sys)
            df = df.merge(
                indicator_df[["tic", "date", indicator]], on=["tic", "date"], how="left"
            )

        df = df.sort_values(by=["date", "tic"])

        return df
    
    def add_cov_matrix(self, data):
        df=data.copy()
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
        return df

    
    def data_split(self,df, start, end, target_date_col="date"):
        """
        split the dataset into training or testing using date
        :param data: (df) pandas dataframe, start, end
        :return: (df) pandas dataframe
        """
        data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
        data = data.sort_values([target_date_col, "tic"], ignore_index=True)
        data.index = data[target_date_col].factorize()[0]
        return data

    def initiate_data_transformation(self,data_path):
        logging.info("Data transformation initiated")
        logging.info("Reading the data")
        df=pd.read_csv(data_path)
        
        df = self.add_tech(df, self.transformation_config.INDICATORS)
        df = df.ffill().bfill()

        df = self.add_cov_matrix(df)
        logging.info("Covariance matrix added")
        logging.info(df.shape)

        hist_vol=[]
        for i in range(len(df['return_list'])):
            returns = df['return_list'].values[i].std()
            hist_vol.append(returns)
        hist_vol= np.array(hist_vol)
        # print(hist_vol.shape)
        # print(hist_vol)
        hist_vol= pd.DataFrame(hist_vol, df['date'])
        logging.info(hist_vol.shape)

        TRAIN_START_DATE = '2011-01-01'
        TRAIN_END_DATE = '2021-12-31'

        # TRAIN_END_DATE = '2012-12-01'

        Val_START_DATE = '2022-01-01'
        VAL_END_DATE =  '2022-12-31'
        TRADE_START_DATE = '2023-01-01'
        TRADE_END_DATE = '2025-02-28'
        # print(df[30:])
        # hist_vol = hist_vol.reset_index(drop=True)

        train = self.data_split(df, TRAIN_START_DATE,TRAIN_END_DATE)
        hist_vol_train = hist_vol[TRAIN_START_DATE : TRAIN_END_DATE]

        val = self.data_split(df, Val_START_DATE, VAL_END_DATE)
        hist_vol_val=hist_vol[Val_START_DATE :VAL_END_DATE]

        full_train = self.data_split(df, TRAIN_START_DATE, VAL_END_DATE)
        hist_vol_full_train= hist_vol[TRAIN_START_DATE :VAL_END_DATE]


        # full_train = data_split(df, TRAIN_START_DATE,TRAIN_END_DATE)
        # hist_vol_full_train= hist_vol[TRAIN_START_DATE :TRAIN_END_DATE]

        trade = self.data_split(df, TRADE_START_DATE,TRADE_END_DATE)
        hist_vol_trade= hist_vol[TRADE_START_DATE  : TRADE_END_DATE]

        logging.info(full_train.shape)

        # full_train.to_csv(self.transformation_config.train_data_path, index=False, header=True)
        # trade.to_csv(self.transformation_config.test_data_path, index=False, header=True)
        return (
            full_train, hist_vol_full_train, train, hist_vol_train, val, hist_vol_val, trade, hist_vol_trade
        )
