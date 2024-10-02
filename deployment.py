import shutil
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 


####################function for deployment
def deploy_model():

    logger.info('copying artefact to prod directory...')
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')
    score_path = os.path.join(config['output_model_path'], 'latestscore.txt')
    ingested_path = os.path.join(dataset_csv_path, 'ingestedfiles.txt')
    artefact_source = [model_path, score_path, ingested_path]

    if not os.path.isdir(prod_deployment_path):
        os.mkdir(prod_deployment_path)

    target_model = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    target_score = os.path.join(prod_deployment_path, 'latestscore.txt')
    target_ingested = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
    artefact_prod = [target_model, target_score, target_ingested]

    for source, target in zip(artefact_source, artefact_prod):
        shutil.copy(source, target)
        
    logger.info('complete...')
        
        
if __name__ == '__main__':
    deploy_model()
