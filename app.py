from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
from diagnostics import dataframe_summary, execution_time, missing_data, model_predictions, outdated_packages_list
import json
import os
import logging

from scoring import score_model


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    logger.info('running predict')
    data_path = request.get_json()['filepath']

    df = pd.read_csv(data_path)
    y_pred = model_predictions(df)
    return jsonify({"prediction": str(y_pred)}) # return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    #check the score of the deployed model
    score = score_model(os.path.join(dataset_csv_path, "finaldata.csv"), prod_deployment_path)
    return jsonify({"score": score}) #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():        
    #check means, medians, and modes for each column
    logger.info('running stats2')
    datasrc = os.path.join(dataset_csv_path, 'finaldata.csv') 
    col_stats = dataframe_summary(datasrc)
    return jsonify(col_stats)#return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    #check timing and percent NA values
    logger.info('running stats3')
    test_data = os.path.join(test_data_path, 'testdata.csv')
    missing = missing_data(test_data)
    time_check = execution_time()
    outdated = outdated_packages_list()
    diagnosis = {'missing': missing, 'time_check': time_check, 'outdated': outdated}
    return jsonify(diagnosis) #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
