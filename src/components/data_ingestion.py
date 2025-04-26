import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.components.yahoo_downloader import YahooDownloader,Tickers

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self,start_date='2011-01-01', end_date='2025-04-26'):
        logging.info("Entered the data initiation method")
        try:
            # Download and save the data in a pandas DataFrame:

            # index= ['sensex_ticker', 'Dow_30', 'dax_30', 'nikkei_top30_symbols', 'FTSE_top30', 'twse_top30', 'hang_seng_symbols', 'brazil_tickers', 'ibex35_tickers', 'bist100_top30_tickers' ]
            df=YahooDownloader(start_date = start_date,
                     end_date = end_date,
                     ticker_list = Tickers.sensex_ticker).fetch_data()
            logging.info('Read the data as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            return self.ingestion_config.raw_data_path 

        except Exception as e:
            raise CustomException(e,sys)
        
