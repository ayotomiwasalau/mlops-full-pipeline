import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from diagnostics import model_predictions



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 





##############Function for reporting
def report_model(outputmodelpath, test_path):
    #calculate a confusion matrix using the test data and the deployed model
    
    test_df = pd.read_csv(test_path)
    y_true = test_df['exited'].values
    y_pred = model_predictions(test_df)

    confusion = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(confusion, annot=True, cmap='Reds')

    ax.set_title('Confusion matrix')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values')

    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])

    plt.savefig(os.path.join(outputmodelpath, 'confusionmatrix.png'))


    

if __name__ == '__main__':
    outputmodelpath = config['output_model_path']
    test_path = os.path.join(test_data_path, 'testdata.csv')
    report_model(outputmodelpath, test_path)
