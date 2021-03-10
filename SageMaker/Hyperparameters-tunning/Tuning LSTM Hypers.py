# MODEL

import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, 2**embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(2**embedding_dim, 2**hidden_dim)
        self.dense = nn.Linear(in_features=2**hidden_dim, out_features=1)
        self.sig = nn.Sigmoid()
        
        self.word_dict = None

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        x = x.t()
        lengths = x[0,:]
        reviews = x[1:,:]
        embeds = self.embedding(reviews)
        lstm_out, _ = self.lstm(embeds)
        out = self.dense(lstm_out)
        out = out[lengths - 1, range(len(lengths))]
        return self.sig(out.squeeze())

import torch.optim as optim
from train.model import LSTMClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(5, 5, 5000).to(device)
optimizer = optim.Adam(model.parameters())
loss_fn = torch.nn.BCELoss()


from sagemaker.pytorch import PyTorch

estimator = PyTorch(entry_point="train.py",
                    source_dir="train",
                    role=role,
                    framework_version='0.4.0',
                    train_instance_count=1,
                    train_instance_type='ml.p2.xlarge',
                    hyperparameters={
                        'epochs': 20,
                        'hidden_dim': 5,
                        'embedding_dim': 5
                    })


from sagemaker.tuner import IntegerParameter, HyperparameterTuner, ContinuousParameter, CategoricalParameter
from time import strftime, gmtime

job_name = "imageclassif-job-{}".format(strftime("%d-%H-%M-%S", gmtime()))

hyperparameter_ranges = {'learning_rate': ContinuousParameter(0.0001, 0.1),
                         'mini_batch_size': IntegerParameter(32, 512),
                         'embedding_dim': IntegerParameter(5, 8),
                         'hidden_dim' : IntegerParameter(5, 8),
                         'optimizer': CategoricalParameter(['sgd', 'adam', 'rmsprop', 'nag'])}

#objective_metric_name = 'validation:accuracy'

tuner = HyperparameterTuner(estimator=estimator, 
                            objective_metric_name='validation:loss', 
                            hyperparameter_ranges=hyperparameter_ranges,
                            objective_type='Minimize',
                            metric_definitions = [{'Name': 'validation:loss',
                                                   'Regex': 'loss (\S+)'}],
                            max_jobs=20, 
                            max_parallel_jobs=1,
                            early_stopping_type='Auto')

s3_input_train = sagemaker.s3_input(s3_data=train_location, content_type='csv')
s3_input_validation = sagemaker.s3_input(s3_data=test_location, content_type='csv')

tuner.fit({'train': s3_input_train, 'validation': s3_input_validation}, job_name=job_name)
tuner.wait()


estimator = sagemaker.estimator.Estimator.attach(tuner.best_training_job())
estimator.fit({'training': train_location, 'validation' : test_location})
