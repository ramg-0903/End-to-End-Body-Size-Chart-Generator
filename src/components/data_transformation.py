import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object


class DataTransformation:
    def __init__(self , initial_data):
        self.data_paths = initial_data[0]
        self.data_bits = initial_data[1]

    def get_data_transformer_object(self , bit):
        
        try:
            
            numerical_columns = {
                0 : ['wai_gi','hip_gi','ank_gi','kne_gi','thi_gi','cal_gi'],
                1 : ['sho_gi','che_gi','wai_gi','nav_gi']
            }

            columns = numerical_columns[bit]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )


            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,columns),
                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self):

        try:

            for i,name in enumerate(self.data_paths):

                train_df=pd.read_csv(self.data_paths[name])
                logging.info(f"Starting Pre-Processing! file - {name}")

                preprocessing_obj=self.get_data_transformer_object(self.data_bits[i])

                logging.info("Removing outliers from training data")
                train_df = self.remove_outliers(train_df)
                train_df.to_csv(self.data_paths[name], index=False) #Updating the csv files.
            
                logging.info(
                    f"Applying preprocessing object on training dataframe"
                )
                
                train_arr=preprocessing_obj.fit_transform(train_df)
                np_path = f'artifacts/scaled_values/{name}.npy'

                np.save(np_path, train_arr)     

                save_object(
                    file_path=f"artifacts/pickle_files/{name}.pkl",
                    obj=preprocessing_obj
                )

                logging.info(f"Successfully Pickle Dumped File - {name}")

        except Exception as e:
            raise CustomException(e,sys)

    
    def remove_outliers(self, df, threshold=1.5):
        try:
            # Iterate over each column in the DataFrame
            for col in df.columns:
                # Calculate IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                # Determine outlier thresholds
                lower_bound = Q1 - (threshold * IQR)
                upper_bound = Q3 + (threshold * IQR)

                # Log all outliers at once
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if not outliers.empty:
                    logging.info(f"Outliers detected in column '{col}':\n{outliers[[col]].to_string(index=True)}")

                # Remove outliers
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

            logging.info(f"Outliers removed. New shape: {df.shape}")
            return df

        except Exception as e:
            logging.error(f"Error in remove_outliers: {e}")
            raise CustomException(e, sys)



