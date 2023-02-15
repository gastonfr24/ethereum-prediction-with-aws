import boto3
import logging

def lambda_handler(event, context):
        try:
            client = boto3.client('sagemaker')
            
            #wish to get current status of instance
            status = client.describe_notebook_instance(NotebookInstanceName='EthereumPrediction')
            
            #Start the instance
            
            client.start_notebook_instance(NotebookInstanceName='EthereumPrediction')
            return 'se ha iniciado la instancia sagemaker'
        except Exception as e:
            return 'error al contactar con sagemaker: {}'.format(str(e))
        
