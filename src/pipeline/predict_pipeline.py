import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self, sho_gi: float,che_gi: str, wai_gi: float, nav_gi: float , gender:str):
        
        self.gender=gender
        self.sho_gi = sho_gi
        self.che_gi = che_gi
        self.wai_gi = wai_gi
        self.nav_gi = nav_gi
        
    

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "sho_gi":[self.sho_gi],
                "che_gi":[self.che_gi],
                "wai_gi":[self.wai_gi],
                "nav_gi":[self.nav_gi]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
