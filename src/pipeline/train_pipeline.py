from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__=="__main__":
    obj=DataIngestion()
    data_path=obj.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    full_train, hist_vol_full_train, train, hist_vol_train, val, hist_vol_val, trade, hist_vol_trade=data_transformation.initiate_data_transformation(data_path)

    model_trainer=ModelTrainer()
    model_trainer.initiate_model_trainer(train, hist_vol_train, val, hist_vol_val,full_train, hist_vol_full_train,  trade, hist_vol_trade)