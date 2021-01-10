split_frac = 0.8

## split data into training, validation, and test data (features and labels, x and y)

train_x = features[:int(len(features)*split_frac),:]
valid_x = features[len(train_x) : len(train_x) + int((len(features) - len(train_x))/2), :]
test_x = features[len(train_x)+len(valid_x):,:]
## print out the shapes of your resultant feature data
print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))

train_y = encoded_labels[:int(len(features)*split_frac)]
valid_y = encoded_labels[len(train_y) : len(train_y) + int((len(features) - len(train_y))/2)]
test_y = encoded_labels[len(train_y)+len(valid_y):]
