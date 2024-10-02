

import apicalls
import training
import ingestion
import scoring
import deployment
import diagnostics
import reporting
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():

    with open("config.json", "r") as f:
        config = json.load(f)

    input_folder_path = config["input_folder_path"]
    prod_deployment_path = config['prod_deployment_path']
    output_folder_path = config['output_folder_path']
    new_data_path = os.path.join(output_folder_path, "finaldata.csv")


    cwd_ingestedfiles = os.path.join(prod_deployment_path, "ingestedfiles.txt")
    cwd_latestscore = os.path.join(prod_deployment_path, "latestscore.txt")

    ingested_files = []
    with open(cwd_ingestedfiles, "r") as report_data_file:
        for entry in report_data_file:
            ingested_files.append(entry.rstrip())

    new_data_files: list = []
    for file in os.listdir(input_folder_path):
        if file[-3:] == 'csv':
            if file not in ingested_files:
                new_data_files.append(file)     
            else:
                pass
                  
    
    if new_data_files:
        ingestion.merge_multiple_dataframe()
        with open(cwd_ingestedfiles, 'w') as f:
            for file in new_data_files:
                f.write(file + '\n') 
            f.close()
    else:
        logger.info('No new data files found...')
        exit(0)


    ### scoring
    with open(cwd_latestscore, 'r') as f:
        deployed_score = float(f.read())

    new_score = scoring.score_model(new_data_path, prod_deployment_path)

    if new_score < deployed_score:
        logger.info("Model drift has occurred")
    else:
        logger.info("No model drift has occurred")
        exit(0)

    ### retraining
    logger.info("retraining model due to model drift")
    training.train_model(output_folder_path, prod_deployment_path)
    logger.info("completing retraining")

    ### redeployment
    logger.info("retraining model due to model drift")
    deployment.deploy_model()
    logger.info("completing retraining")


    logging.info("Running diagnostics and reporting")

    with open("config.json", "r") as f:
        config = json.load(f)
    config['output_model_path'] = "models"

    with open("config.json", "w") as f:
        f.write(json.dumps(config))

    output_model_path = config['output_model_path']

    # Run diagnostics.py and reporting.py for the re-deployed model
    reporting.report_model(output_model_path, new_data_path)

    apicalls.apicalls(output_model_path, new_data_path)




if __name__ == '__main__':
    main()



