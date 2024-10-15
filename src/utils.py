# utils is having all the common things which we tring to import or use whic entire project will be using

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV


import dill

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(xtrain, ytrain, xtest, ytest, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(xtrain, ytrain)

            model.set_params(**gs.best_params_)
            model.fit(xtrain, ytrain)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(xtrain)

            y_test_pred = model.predict(xtest)

            train_model_score = r2_score(ytrain, y_train_pred)

            test_model_score = r2_score(ytest, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)