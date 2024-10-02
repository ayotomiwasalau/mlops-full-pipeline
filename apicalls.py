import logging
import requests
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with open("config.json", "r") as f:
        config = json.load(f)

model_path = config['prod_deployment_path']
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:5000/"

#Call each API endpoint and store the responses

def apicalls(model_path, test_data_path):

    prediction = requests.post(
        f'{URL}/prediction',
        json={'filepath': test_data_path}
    ).text

    logging.info("Get request /scoring")
    scoring = requests.get(f'{URL}/scoring').text

    logging.info("Get request /summarystats")
    summarystats = requests.get(f'{URL}/summarystats').text

    logging.info("Get request /diagnostics")
    diagnostics = requests.get(f'{URL}/diagnostics').text

    #combine all API responses
    logging.info("Generating report text file")
    with open(os.path.join(model_path, 'apireturns.txt'), 'w') as file:
        file.write('Ingested Data\n\n')
        file.write('Statistics Summary\n')
        file.write(summarystats)
        file.write('\nDiagnostics Summary\n')
        file.write(diagnostics)
        file.write('\n\nTest Data\n\n')
        file.write('Model Predictions\n')
        file.write(prediction)
        file.write('\nModel Score\n')
        file.write(scoring)
        file.close()

if __name__ == '__main__':
    apicalls(model_path, test_data_path)
