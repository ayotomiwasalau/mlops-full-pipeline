from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import logging
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
output_model_path = config['output_model_path']


#################Function for training the model
def train_model(dataset_csv_path, output_model_path):
    
    logger.info("Training model...")

    model_path = os.path.join(output_model_path, 'trainedmodel.pkl') 
    current_dir_dataset_csv_path = os.path.join(dataset_csv_path, 'finaldata.csv')
    #logistic model regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                     n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)

    
    
    #fit the logistic regression to data

    X = pd.read_csv(current_dir_dataset_csv_path).iloc[:, 1:]
    y = X.pop('exited').values
    model.fit(X, y)

    logger.info("Model trained...")
    
    #save trained model in a file called trainedmodel.pkl

    try:
        pickle.dump(model, open(model_path, 'wb'))
        logger.info(f'model saved in {model_path}')
    except Exception as e:
        raise(e)

if __name__ == '__main__':
    train_model(dataset_csv_path, output_model_path)