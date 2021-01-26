import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()
        
        # define all layers
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_size)
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        # flatten image
        x = self.relu(self.fc1(x), 0.1) # negative slope of 0.1
        x = self.relu(self.fc2(x), 0.1)
        x = self.fc3(x)
        # pass x through all layers
        # apply leaky relu activation to all hidden layers

        return x
