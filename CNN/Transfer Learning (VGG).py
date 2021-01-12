from torchvision import datasets, models, transforms

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()




# Load the pretrained model from pytorch
vgg16 = models.vgg16(pretrained=True)




# print out the model structure
print(vgg16)
# the last line of the printed output, usually Linear classifier,
# will tell the number of out and in features
# OR access it by
print(vgg16.classifier[6].in_features) 
print(vgg16.classifier[6].out_features)
# 6 is the number of that last line





# Freeze training for all "features" layers
for param in vgg16.features.parameters():
    param.requires_grad = False




# Replace the last layer
import torch.nn as nn

n_inputs = vgg16.classifier[6].in_features
# add last linear layer (n_inputs -> 5 flower classes)
# new layers automatically have requires_grad = True
last_layer = nn.Linear(n_inputs, len(classes))
vgg16.classifier[6] = last_layer
# if GPU is available, move the model to GPU
if train_on_gpu:
    vgg16.cuda()
# check to see that your last layer produces the expected number of outputs
print(vgg16.classifier[6].out_features)
#print(vgg16)





