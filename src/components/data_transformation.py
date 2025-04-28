from datetime import date
from stockstats import StockDataFrame as Sdf
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
        self.df=pd.read_csv(data_path)
        
        self.df = self.add_tech(self.df, self.transformation_config.INDICATORS)
        self.df = self.df.ffill().bfill()

        self.df = self.add_cov_matrix(self.df)
        logging.info("Covariance matrix added")
        logging.info(self.df.shape)

        self.hist_vol=[]
        for i in range(len(self.df['return_list'])):
            returns = self.df['return_list'].values[i].std()
            self.hist_vol.append(returns)
        self.hist_vol= np.array(self.hist_vol)
        self.hist_vol= pd.DataFrame(self.hist_vol, self.df['date'])
        logging.info(self.hist_vol.shape)

    def get_train_val_data(self,TRAIN_START_DATE = '2011-01-01',TRAIN_END_DATE = '2012-04-01',VAL_START_DATE = '2012-04-01', VAL_END_DATE = '2013-01-01'):
        train = self.data_split(self.df, TRAIN_START_DATE,TRAIN_END_DATE)
        hist_vol_train = self.hist_vol[TRAIN_START_DATE : TRAIN_END_DATE]

        val = self.data_split(self.df, VAL_START_DATE, VAL_END_DATE)
        hist_vol_val=self.hist_vol[VAL_START_DATE :VAL_END_DATE]

        full_train = self.data_split(self.df, TRAIN_START_DATE,VAL_END_DATE)
        hist_vol_full_train= self.hist_vol[TRAIN_START_DATE :VAL_END_DATE]

        logging.info(full_train.shape)

        return (
            full_train, hist_vol_full_train, train, hist_vol_train, val, hist_vol_val
        )
    
    def get_trade_data(self,TRADE_START_DATE = '2025-01-01',TRADE_END_DATE = date.today().strftime("%Y-%m-%d")):
        trade = self.data_split(self.df, TRADE_START_DATE,TRADE_END_DATE)
        hist_vol_trade= self.hist_vol[TRADE_START_DATE  : TRADE_END_DATE]
        logging.info(trade.shape)
        return trade, hist_vol_trade
