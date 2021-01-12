validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
  test_set.data,
  test_set.target,
  every_n_steps=50,
  metrics=validation_metrics,
  early_stopping_metric="loss",
  early_stopping_metric_minimize=True,
  early_stopping_rounds=200)


classifier = tf.contrib.learn.DNNClassifier(
  feature_columns=feature_columns,
  hidden_units=[10, 20, 10],
  n_classes=3,
  model_dir="/tmp/iris_model",
  config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))

classifier.fit(x=training_set.data,
           y=training_set.target,
           steps=2000,
           monitors=[validation_monitor])
