
import pandas as pd
import numpy as np
import timeit
import os
import pickle
import json
import logging
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open('config.json','r') as f:
    config = json.load(f) 

##################Function to get model predictions
def model_predictions(data):
    logger.info('calculate model predictions')
    #read the deployed model and a test dataset, calculate predictions

    production_path = os.path.join(config['prod_deployment_path'])
    model_path = os.path.join(production_path, 'trainedmodel.pkl')

    model = pickle.load(open(model_path, 'rb'))
    X = data.iloc[:, 1:-1]
    predictions = list(model.predict(X))
    return predictions #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary(datasource):
    #calculate summary statistics here
    logger.info('calculate statistics on the data')
    data = pd.read_csv(datasource)

    X = data.iloc[:, 1:-1]
    means = X.mean()
    medians = X.median()
    std_var = X.std()

    col_stats = {}
    for col in X.columns:
        col_stats[col] = {'mean': means[col], 'median': medians[col], 'std_dev': std_var[col]}
    return col_stats #return value should be a list containing all summary statistics

def missing_data(test_data):
    #check for missing data
    logger.info('check for missing data')
    data = pd.read_csv(test_data)
    missing_values = data.isnull().sum()
    n_count = data.shape[0]
    missing_percentage = (missing_values/n_count)*100
    return list(missing_percentage)#return the number of missing values

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    ingestion_time = []
    training_time = []
    for i in range(20):  
        logging.info(f"testing timing {i}")      
        try:
            ingeststarttime = timeit.default_timer()
            # Capture output and errors from ingestion.py
            result = subprocess.run(['python', 'ingestion.py'], capture_output=False)
            if result.stderr:
                logging.error(result.stderr)  # Log any errors
            ingestion_timing = timeit.default_timer() - ingeststarttime
            ingestion_time.append(ingestion_timing)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error occurred while running ingestion.py: {e}")

        # timing training
        try:
            trainingstarttime = timeit.default_timer()
            # Capture output and errors from training.py
            result = subprocess.run(['python3', 'training.py'], capture_output=False)
            logging.info(result.stdout)  # Log the standard output
            if result.stderr:
                logging.error(result.stderr)  # Log any errors
            training_timing = timeit.default_timer() - trainingstarttime
            training_time.append(training_timing)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error occurred while running training.py: {e}")

    return [np.mean(ingestion_time), np.mean(training_time)] #return a list of 2 timing values in seconds

##################Function to check dependencies

def outdated_packages_list():
    logging.info("Checking outdated dependencies")
    
    # Read current dependencies from requirements.txt
    with open('requirements.txt', 'r') as req_file:
        requirements = {line.split('==')[0].strip(): line.split('==')[1].strip() 
                        for line in req_file if '==' in line}

    # Get outdated dependencies using pip
    try:
        # Use JSON format to get structured output
        outdated_dep = subprocess.check_output(['pip', 'list', '--outdated', '--format=json'], text=True)
        
        # Parse JSON output to extract package names and their latest versions
        import json
        outdated_list = json.loads(outdated_dep)
        outdated_dict = {pkg['name']: pkg['latest_version'] for pkg in outdated_list}

    except subprocess.CalledProcessError as e:
        logging.error(f"Error checking outdated packages: {e}")
        return []

    # Create a list of records with current and latest versions
    result = []
    for module, current in requirements.items():
        latest = outdated_dict.get(module, current)
        result.append({'module': module, 'current': current, 'latest': latest})

    return result


if __name__ == '__main__':
    ##################Load config.json and get environment variables

    dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv') 
    test_data_path = os.path.join(config['test_data_path'], 'testdata.csv') 

    data = pd.read_csv(test_data_path)
    model_predictions(data)
    dataframe_summary(dataset_csv_path)
    missing_data(test_data_path)
    execution_time()
    outdated_packages_list()





    
