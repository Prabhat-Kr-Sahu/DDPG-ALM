import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.components.yahoo_downloader import YahooDownloader,Tickers
# from src.components.data_transformation import DataTransformation,DataTransformationConfig
# from src.components.model_trainer import ModelTrainer,ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data initiation method")
        try:
            # indices= ['sensex_ticker', 'Dow_30', 'dax_30', 'nikkei_top30_symbols', 'FTSE_top30', 'twse_top30', 'hang_seng_symbols', 'brazil_tickers', 'ibex35_tickers', 'bist100_top30_tickers' ]
            # Download and save the data in a pandas DataFrame:
            df=YahooDownloader(start_date = '2011-01-01',
                     end_date = '2025-02-28',
                     ticker_list = Tickers.sensex_ticker).fetch_data()
            logging.info('Read the data as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            # train_df,test_df=train_test_split(df,test_size=0.2,random_state=18)
            # logging.info('Train-test split successful')
            # train_df.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            # test_df.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            # logging.info('Data ingestion is completed')

            # return (
            #     self.ingestion_config.train_data_path,
            #     self.ingestion_config.test_data_path
            # )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()
    # train_path,test_path=obj.initiate_data_ingestion()

    # data_transformation=DataTransformation()
    # train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_path,test_path)

    # model_trainer=ModelTrainer()