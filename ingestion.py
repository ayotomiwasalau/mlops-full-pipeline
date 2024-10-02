import pandas as pd
import numpy as np
import os
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f)  

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe(finaldata: str='finaldata.csv'):

    dataset_list = []
    #check for datasets, compile them together, and write to an output file
    record_list =  os.path.join(output_folder_path, 'ingestedfiles.txt')
    filenames = os.listdir(input_folder_path)  

    logger.info("Ingesting datasets and merging...")
    
    if len(filenames) != 0: 
        with open(record_list, 'w') as f:  
            for each_filename in filenames:
                if each_filename[-3:] == 'csv':         
                    df = pd.read_csv(os.path.join(input_folder_path,each_filename))
                    dataset_list.append(df)
                    f.write(each_filename + '\n')
    else:
        logger.info('No data files in folder')
    dataset = pd.concat(dataset_list)
    dataset = dataset.drop_duplicates()
    dataset.to_csv(os.path.join(output_folder_path, finaldata), index=False)

    logger.info(f"completed...saved to {output_folder_path}")



if __name__ == '__main__':
    merge_multiple_dataframe()
