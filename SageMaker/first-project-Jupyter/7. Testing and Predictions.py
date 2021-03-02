# TODO: Create a transformer object from the trained model. Using an instance count of 1 and an instance type of ml.m4.xlarge
#       should be more than enough.
xgb_transformer = xgb.transformer(instance_count=1, instance_type='ml.m4.xlarge')


# TODO: Start the transform job. Make sure to specify the content type and the split type of the test data.
xgb_transformer.transform(test_location, content_type='text/csv', split_type='Line')

# Currently the transform job is running but it is doing so in the background. Since we wish to wait until the transform job is done and we would like a bit of feedback we can run the wait() method.
xgb_transformer.wait()


# Now the transform job has executed and the result, the estimated sentiment of each review,
# has been saved on S3. Since we would rather work on this file locally
# we can perform a bit of notebook magic to copy the file to the data_dir.
!aws s3 cp --recursive $xgb_transformer.output_path $data_dir



# Predictions and accuracy


predictions = pd.read_csv(os.path.join(data_dir, 'test.csv.out'), header=None)
predictions = [round(num) for num in predictions.squeeze().values]

from sklearn.metrics import accuracy_score
accuracy_score(test_y, predictions)



#  Optional Clean Up

# First we will remove all of the files contained in the data_dir directory
!rm $data_dir/*

# And then we delete the directory itself
!rmdir $data_dir

# Similarly we will remove the files in the cache_dir directory and the directory itself
!rm $cache_dir/*
!rmdir $cache_dir
