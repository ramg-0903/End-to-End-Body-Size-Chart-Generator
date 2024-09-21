import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd


from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainer
from src.components.model_predictor import ModelPredictor
from src.components.data_updation import DataUpdation

    

class DataIngestion:
    def __init__(self):
        self.data = []

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion Component")
        try:

            data_paths = {}
            data_bits = [] # used to specify if it's lower or upper body

            data_paths['men_lower'] ="artifacts/updated_data/men_lower.csv"
            data_bits.append(0)
            data_paths['men_upper'] ="artifacts/updated_data/men_upper.csv"
            data_bits.append(1)
            data_paths['women_lower'] ="artifacts/updated_data/women_lower.csv"
            data_bits.append(0)
            data_paths['women_upper'] ="artifacts/updated_data/women_upper.csv"
            data_bits.append(1)

            self.data.append(data_paths)
            self.data.append(data_bits)

            logging.info("Ingestion of the data is completed")

            return self.data

            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":

    ingestion_obj=DataIngestion()
    initial_data=ingestion_obj.initiate_data_ingestion()

    data_transformation=DataTransformation(initial_data)
    data_transformation.initiate_data_transformation()

    modeltrainer=ModelTrainer()
    modeltrainer.initiate_model_trainer()

    modeltrainer=ModelPredictor()
    modeltrainer.initiate_model_trainer()

    logging.info("Completed Everything!")

    


