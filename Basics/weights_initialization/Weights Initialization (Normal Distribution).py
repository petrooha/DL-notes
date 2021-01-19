import torch.nn as nn
import torch.nn.functional as F

# define the NN architecture
class Net(nn.Module):
    def __init__(self, hidden_1=256, hidden_2=128, constant_weight=None):
        super(Net, self).__init__()
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        # linear layer (hidden_1 -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # linear layer (hidden_2 -> 10)
        self.fc3 = nn.Linear(hidden_2, 10)
        # dropout layer (p=0.2)
        self.dropout = nn.Dropout(0.2)

        """In the case below, we look at every layer/module in our model.
        If it is a Linear layer (as all three layers are for this MLP),
        then we initialize those layer weights to be a constant_weight
        with bias=0 using the following code:
        """
        
        # initialize the weights to a specified, constant value
        if(constant_weight is not None):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, constant_weight)
                    nn.init.constant_(m.bias, 0)






    # Define a function that assigns weights by the type of network layer, then
    # Apply those weights to an initialized model using model.apply(fn),
    # which applies a function to each model layer.

    def weights_init_normal(m):
        '''Takes in a module and initializes all linear layers with weight
           values taken from a normal distribution.'''
        
        classname = m.__class__.__name__
        # for every Linear layer in a model
        if classname.find('Linear') != -1:
            n = m.in_features
            y = 1.0/np.sqrt(n)
            # m.weight.data shoud be taken from a normal distribution
            m.weight.data.normal_(0.0, y)
            # m.bias.data should be 0
            m.bias.data.fill_(0)
        
    # create a new model with these weights
    model_uniform = Net()
    model_uniform.apply(weights_init_uniform)






