import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.Components.Data_Transformation import DataTransformation
from src.Components.Data_Transformation import DataTransformationConfig
from src.Components.Model_Trainer import ModelTrainer
from src.Components.Model_Trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
#inputs giving to Data - Ingestion components
    train_data_path: str = os.path.join('artifact', "train.csv")
    test_data_path: str = os.path.join('artifact', "test.csv")
    raw_data_path: str = os.path.join('artifact', "data.csv")
# now data ingestion components knows where to save the train, test and raw data path

class DataIngestion:
    def __init__(self):
# as soon as I run this class the above train, test, raw paths will be saved to ingestion_config variable
        self.ingestion_config = DataIngestionConfig()
    
# below function will help to read the data from any kind of databases
    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion method or component")
        try:
            df = pd.read_csv('Notebooks\Data\stud.csv') # we can read the data from any database here I am using the local database.
            logging.info("Read the Dataset as Dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
            
            logging.info("Train Test Split initiated")
            
            train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Ingestion of the Data is Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))