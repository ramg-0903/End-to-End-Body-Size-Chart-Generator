import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd


class DataUpdation:
    def __init__(self):
        self.ingestion_config=""
    
    def initiate_update(self):
        
        logging.info("Entered the data updation phase")

        try:
            pass

        except Exception as e:
            raise CustomException(e,sys)