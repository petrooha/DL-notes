# Start endpoint
xgb_predictor = xgb.deploy(initial_instance_count = 1, instance_type = 'ml.m4.xlarge')

# Tell endpoint what type of data
from sagemaker.predictor import csv_serializer

xgb_predictor.content_type = 'text/csv'
xgb_predictor.serializer = csv_serializer


# Pass "body" to endpoint and get responce back
response = runtime.invoke_endpoint(EndpointName = xgb_predictor.endpoint,   # The name of the endpoint we created
                                       ContentType = 'text/csv',                     # The data format that is expected
                                       Body = ','.join([str(val) for val in test_bow]).encode('utf-8'))

response = response['Body'].read().decode('utf-8')
print(response)

# Shut off endpoint
xgb_predictor.delete_endpoint()
