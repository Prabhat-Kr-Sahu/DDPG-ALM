from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def run_pipeline():
    obj=DataIngestion()
    data_path=obj.initiate_data_ingestion()

    TRAIN_START_DATE = '2011-01-01'
    TRAIN_END_DATE = '2012-04-01'
    VAL_START_DATE = '2012-04-01'
    VAL_END_DATE = '2013-01-01'
    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(data_path)
    full_train, hist_vol_full_train, train, hist_vol_train, val, hist_vol_val = data_transformation.get_train_val_data(TRAIN_START_DATE, TRAIN_END_DATE, VAL_START_DATE, VAL_END_DATE)

    model_trainer=ModelTrainer()
    model_trainer.initiate_model_trainer(train, hist_vol_train, val, hist_vol_val,full_train, hist_vol_full_train)