import os
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object , load_object


class ModelPredictor:
    def __init__(self):
        self.training_paths = "artifacts/updated_data"
        self.preprocessor_paths = "artifacts/pickle_files"
        self.model_path = "artifacts/logit_model"


    def initiate_model_trainer(self):

        try:
            
            data_paths = os.listdir(self.training_paths)

            for training_data_path in data_paths:

                file_name = os.path.splitext(training_data_path)[0]

                logging.info(f"Starting Model Training for {file_name}")

                train_path = os.path.join(self.training_paths , training_data_path)
                df=pd.read_csv(train_path)

                X = df.drop(columns=['labels'])
                y = df['labels']
                
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

                preprocessor_path=os.path.join(self.preprocessor_paths , f"{file_name}.pkl")
                preprocessor=load_object(file_path=preprocessor_path)
                data_scaled=preprocessor.transform(X)
                test_data_Scaled = preprocessor.transform(X_test)

                model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=100, random_state=42)

                model.fit(data_scaled, y)
                y_pred = model.predict(test_data_Scaled)
                save_object(
                    file_path=os.path.join(self.model_path , file_name)+ '.pkl',
                    obj=model
                )
                logging.info(f"Model Trained for file {file_name} - Accuracy ({accuracy_score(y_test, y_pred)})")


        except Exception as e:
            raise CustomException(e, sys)

