import math
def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    # TODO: Implement batching
    batches = []
    if len(features) % batch_size == 0:
        r = len(features)/batch_size
    else:
        r = len(features)//batch_size + 1
        
    for i in range(r):
        ii = i*batch_size
        batch = [features[ii:ii+batch_size], labels[ii:ii+batch_size]]
        batches.append(batch)
    
    return batches
