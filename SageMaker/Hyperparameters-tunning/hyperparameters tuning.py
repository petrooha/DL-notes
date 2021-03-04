xgb = sagemaker.estimator.Estimator(container, role, train_instance_count=1, train_instance_type='ml.m4.xlarge',
                                   output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
                                   sagemaker_session=session)

# TODO: Set the XGBoost hyperparameters in the xgb object. Don't forget that in this case we have a binary
#       label so we should be using the 'binary:logistic' objective.
xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        objective='binary:logistic',
                        early_stopping_rounds=10,
                        num_round=200)


# First, make sure to import the relevant objects used to construct the tuner
from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner

# TODO: Create the hyperparameter tuner object

xgb_hyperparameter_tuner = HyperparameterTuner(estimator=xgb, objective_metric_name='validation:rmse', 
                                               objective_type='Minimize', max_jobs=20, max_parallel_jobs=3, 
                                               hyperparameter_ranges = {'max_depth':IntegerParameter(3,12),
                                                                      'eta':ContinuousParameter(0.05, 0.5),
                                                                      'min_child_weight':IntegerParameter(2,8),
                                                                      'subsample':ContinuousParameter(0.5, 0.9),
                                                                      'gamma':ContinuousParameter(0, 10)})



s3_input_train = sagemaker.s3_input(s3_data=train_location, content_type='csv')
s3_input_validation = sagemaker.s3_input(s3_data=val_location, content_type='csv')

xgb_hyperparameter_tuner.fit({'train': s3_input_train, 'validation': s3_input_validation})

xgb_hyperparameter_tuner.wait()


# TODO: Create a new estimator object attached to the best training job found during hyperparameter tuning
xgb_attached = sagemaker.estimator.Estimator.attach(xgb_hyperparameter_tuner.best_training_job())

xgb_transformer = xgb_attached.transformer(instance_count=1, instance_type='ml.m4.xlarge')

# TODO: Start the transform job. Make sure to specify the content type and the split type of the test data.
xgb_transformer.transform(test_location, content_type='text/csv', split_type='Line')

xgb_transformer.wait()





