# helper deconv function
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    ## TODO: Complete this function
    ## create a sequence of transpose + optional batch norm layers
    layers = []
    conv_trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False) 
    layers.append(conv_trans)
    
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    
    def __init__(self, z_size, conv_dim=32):
        super(Generator, self).__init__()

        # complete init function
        self.fc = nn.Linear(z_size, 128*4*4)
        
        self.deconv1 = deconv(128, 64, 4)
        self.deconv2 = deconv(64, 32, 4)
        self.deconv3 = deconv(32, 3, 4, batch_norm=False)
        

    def forward(self, x):
        # complete forward function
        x = self.fc(x)
        x = x.view(-1, 128, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.tanh(self.deconv3(x))
        return x
    
