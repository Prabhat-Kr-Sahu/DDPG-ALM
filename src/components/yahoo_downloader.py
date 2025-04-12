"""Contains methods and classes to collect data from
Yahoo Finance API
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from src.exception import CustomException
from src.logger import logging
import sys
import os

class Tickers:
    Nifty_ticker = ['RELIANCE.NS', 'ASIANPAINT.NS', 'BAJFINANCE.NS', 'HDFCBANK.NS', 'SBIN.NS']
    sensex_ticker = ["ASIANPAINT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BHARTIARTL.NS", "HCLTECH.NS", "HDFCBANK.NS",
                    "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
                    "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBIN.NS", "SUNPHARMA.NS",
                    "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS"]


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


class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data (modified from neofinrl_config.py)
        end_date : str
            end date of the data (modified from neofinrl_config.py)
        ticker_list : list
            a list of stock tickers (modified from neofinrl_config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self, proxy=None, auto_adjust=False) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        num_failures = 0
        for tic in self.ticker_list:
            temp_df = yf.download(
                tic,
                start=self.start_date,
                end=self.end_date,
                proxy=proxy,
                auto_adjust=auto_adjust,
            )
            if temp_df.columns.nlevels != 1:
                temp_df.columns = temp_df.columns.droplevel(1)
            temp_df["tic"] = tic
            if len(temp_df) > 0:
                # data_df = data_df.append(temp_df)
                data_df = pd.concat([data_df, temp_df], axis=0)
            else:
                num_failures = num_failures+ 1
        if num_failures == len(self.ticker_list):
            raise CustomException(ValueError("no data is fetched."),sys)
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        try:
            # convert the column names to standardized names
            data_df.rename(
                columns={
                    "Date": "date",
                    "Adj Close": "adjcp",
                    "Close": "close",
                    "High": "high",
                    "Low": "low",
                    "Volume": "volume",
                    "Open": "open",
                    "tic": "tic",
                },
                inplace=True,
            )

            # use adjusted close price instead of close price
            data_df["close"] = data_df["adjcp"]
            # drop the adjusted close price column
            data_df = data_df.drop(labels="adjcp", axis=1)
        except Exception as e:
            raise CustomException(e, sys)
        # except NotImplementedError:
        #     print("the features are not supported currently")
        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        logging.info("Data downloaded from Yahoo Finance API")
        logging.info("Shape of DataFrame: {}".format(data_df.shape))
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)
        
        return data_df

    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df