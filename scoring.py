from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

output_model_path = config['output_model_path']
test_path = config['test_data_path']
test_data_path = os.path.join(test_path, 'testdata.csv') 



#################Function for model scoring
def score_model(test_data_path, output_model_path):
    #take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #write the result to the latestscore.txt file
    
    model_path = os.path.join(output_model_path, 'trainedmodel.pkl') 
    output_path = os.path.join(output_model_path, 'latestscore.txt')

    test_data = pd.read_csv(test_data_path)
    model = pickle.load(open(model_path, 'rb'))
    X = test_data.iloc[:, 1:]
    y = X.pop('exited').values
    y_pred = model.predict(X)
    f1_score = metrics.f1_score(y, y_pred)
    
    with open(output_path, 'w') as f:
        f.write(str(f1_score) + '\n')
        f.close

    logger.info(f'f1 score saved in {output_path}')
    return f1_score


if __name__ == '__main__':
    score_model(test_data_path, output_model_path)