"""
main purpose to create this is - 
to do feature engineering, data cleaning, data transformation
converting categorical data into numerical / scaling the numerical data.

"""
import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer #to create pipelines 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler # to transform the data

from src.exception import CustomException # to handle the exception
from src.logger import logging # to create a logs
from src.utils import save_object

@dataclass
class DataTransformationConfig: # it will give any path / inputs we require for my data transformation component
    preprocessor_obj_file_path = os.path.join('artifact', 'preprocessor.pkl') #

class DataTransformation: # this should be the input we are giving over here
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self): # this function will help to perform data transformation on input data.
        # this function is responsible for data transformation

        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender", 
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            #creating a pipeline to transform the numerical data along with handeling the outlier data.
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "median")),
                    ("scaler", StandardScaler())
                ]
            )

            # creating a pipeline to transform and handle categorical data.
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"Categorical Columns: {categorical_columns}")
            logging.info(f"Numerical COlumns: {numerical_columns}")

            # below is all in one column transformer with pipeline which is a combination of numerical and categorical pipeline
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
            

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining Preprocessing object")

            preprocessor_obj =self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)###########################

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Saved Preprocessing Object.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)