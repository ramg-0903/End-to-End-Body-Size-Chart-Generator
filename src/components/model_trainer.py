import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object


class ModelTrainer:
    def __init__(self):
        self.model_paths = "artifacts/model_files"
        self.scaled_data_path = "artifacts/scaled_values"
        self.data_path = "artifacts/updated_data"


    def initiate_model_trainer(self):

        try:
            
            data_paths = os.listdir(self.scaled_data_path)

            for path in data_paths:

                file_name = os.path.splitext(path)[0]
                file_path = os.path.join(self.scaled_data_path , path)

                logging.info(f"Preparing for clustering - {file_name}")
                scaled_array = np.load(file_path) 
                n_clusters = 4  # You can adjust the number of clusters as needed
                

                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans.fit(scaled_array)
                cluster_labels = kmeans.predict(scaled_array)
                
                silhouette_avg = silhouette_score(scaled_array, cluster_labels)
                
                logging.info(f"Silhouette Score for K-means clustering: {silhouette_avg} - {file_name}")

                update_path = os.path.join(self.data_path , f"{file_name}.csv")
                df = pd.read_csv(update_path)

                df['labels'] = cluster_labels
                df.to_csv(update_path , index=False)
                logging.info(f"Updated data files and saving model - {file_name}")

                save_object(
                    file_path=os.path.join(self.model_paths , file_name)+"_ml" + '.pkl',
                    obj=kmeans
                )


        except Exception as e:
            raise CustomException(e, sys)


    