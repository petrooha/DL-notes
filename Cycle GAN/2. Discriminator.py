import torch.nn as nn
import torch.nn.functional as F

# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    
    def __init__(self, conv_dim=64):
        super(Discriminator, self).__init__()

        # Define all convolutional layers
        # Should accept an RGB image as input and output a single value
        self.conv1 = conv(3, 32, 4, batch_norm=False)
        self.conv2 = conv(32, 64, 4)
        self.conv3 = conv(64, 128, 4)
        self.conv4 = conv(128, 256, 4)
        self.conv5 = conv(256, 512, 4)
        
        # self.conv6 = conv(512, 1, 4, stride=1, batch_norm=False)
        self.fc = nn.Linear(512*4*4, 1)

    def forward(self, x):
        # define feedforward behavior
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        
        # x = self.conv6(x)
        x = x.view(-1, 512*4*4)
        x = self.fc(x)
        return x
