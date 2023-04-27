import os
import sys
import numpy as np
import pandas as pd
import dill

from src.exception import Custom_Exception
from sklearn.model_selection import train_test_split

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)
    
    except Exception as e:
        raise Custom_Exception(e, sys)
    
def evaluate_model(X,y,models):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,y,test_size=0.2,random_state=42
        )

        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train, y_train) # Train Model

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            


    except Exception as e:
        raise Custom_Exception(e,sys)
