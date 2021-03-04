xgb_predictor = xgb.deploy(initial_instance_count = 1, instance_type = 'ml.m4.xlarge')

from sagemaker.predictor import csv_serializer

# We need to tell the endpoint what format the data we are sending is in so that SageMaker can perform the serialization.
xgb_predictor.content_type = 'text/csv'
xgb_predictor.serializer = csv_serializer


# We split the data into chunks and send each chunk seperately, accumulating the results.

def predict(data, rows=512):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = ''
    for array in split_array:
        predictions = ','.join([predictions, xgb_predictor.predict(array).decode('utf-8')])
    
    return np.fromstring(predictions[1:], sep=',')


test_X = pd.read_csv(os.path.join(data_dir, 'test.csv'), header=None).values

predictions = predict(test_X)
predictions = [round(num) for num in predictions]


from sklearn.metrics import accuracy_score
accuracy_score(test_y, predictions)


xgb_predictor.delete_endpoint()
