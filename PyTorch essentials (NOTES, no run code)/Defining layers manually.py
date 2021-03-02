import torch

def activation(x):
    return 1/(1+torch.exp(-x))

def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)


torch.manual_seed(7)

features = torch.randn((1,5))
hidden = torch.randn((1,3))
weights = torch.randn(features.shape[1], hidden.shape[1])
bias = torch.randn((1,1))

weights2 = torch.randn((hidden.shape[1], 1))
bias2 = torch.randn((1,1))

h_output = activation(torch.mm(features, weights) + bias)
output = activation(torch.mm(h_output, weights2) + bias2)
