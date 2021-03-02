import pandas as pd

# TODO: Split the train_X and train_y arrays into the DataFrames val_X, train_X and val_y, train_y. Make sure that
#       val_X and val_y contain 10 000 entires while train_X and train_y contain the remaining 15 000 entries.
train_X, val_X, train_y, val_y = sklearn.model_selection.train_test_split(train_X, train_y, test_size=0.4)

# Optional split:
"""
val_X = pd.DataFrame(train_X[:10000])
train_X = pd.DataFrame(train_X[10000:])
val_y = pd.DataFrame(train_X[:10000])
train_y = pd.DataFrame(train_X[10000:])
"""


# First we make sure that the local directory in which we'd like to store the training and validation csv files exists.
data_dir = '../data/xgboost'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)



# First, save the test data to test.csv in the data_dir directory. Note that we do not save the associated ground truth
# labels, instead we will use them later to compare with our model output.

pd.DataFrame(test_X).to_csv(os.path.join(data_dir, 'test.csv'), header=False, index=False)

# TODO: Save the training and validation data to train.csv and validation.csv in the data_dir directory.
#       Make sure that the files you create are in the correct format.

pd.concat([val_y, val_X], axis=1).to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)
pd.concat([train_y, train_X], axis=1).to_csv(os.path.join(data_dir, 'validation.csv'), header=False, index=False)

# To save a bit of memory we can set text_X, train_X, val_X, train_y and val_y to None.
test_X = train_X = val_X = train_y = val_y = None



####  TO S3


import sagemaker

session = sagemaker.Session() # Store the current SageMaker session

# S3 prefix (which folder will we use)
prefix = 'sentiment-xgboost'

# TODO: Upload the test.csv, train.csv and validation.csv files which are contained in data_dir to S3 using sess.upload_data().
test_location = session.upload_data(os.path.join(data_dir, 'test.csv'), key_prefix=prefix)
val_location = session.upload_data(os.path.join(data_dir, 'validation.csv'), key_prefix=prefix)
train_location = session.upload_data(os.path.join(data_dir, 'train.csv'), key_prefix=prefix)
