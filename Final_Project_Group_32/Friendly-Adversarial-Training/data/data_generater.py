import os
import tarfile
import pickle
import numpy as np
from collections import defaultdict
from urllib.request import urlretrieve

# Download and extract CIFAR-10 dataset
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
filename = 'cifar-10-python.tar.gz'
if not os.path.exists(filename):
    urlretrieve(url, filename)

with tarfile.open(filename, 'r:gz') as tar:
    tar.extractall()




# Load CIFAR-10 dataset
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


data_batches = [unpickle(f'cifar-10-batches-py/data_batch_{i}') for i in range(1, 6)]
test_batch = unpickle('cifar-10-batches-py/test_batch')


# Create a new dataset with 600 images per label
new_dataset = defaultdict(list)
num_images_per_label = 600

for batch in data_batches:
    for data, label in zip(batch[b'data'], batch[b'labels']):
        if len(new_dataset[label]) < num_images_per_label:
            new_dataset[label].append(data)

# Save the new dataset as small-cifar-10-python.tar.gz
if not os.path.exists('small-cifar-10-batches-py'):
    os.makedirs('small-cifar-10-batches-py')

for i in range(1, 6):
    filename = f'small-cifar-10-batches-py/data_batch_{i}'
    _data = {
        b'data': np.concatenate([new_dataset[j] for j in range(10) if j != i-1]),
        b'labels': np.repeat(np.arange(10), num_images_per_label - (i-1)*100)
    }
    with open(filename, 'wb') as f:
        pickle.dump(_data, f)

_data = {
    b'data': np.concatenate([new_dataset[i] for i in range(10)]),
    b'labels': np.repeat(np.arange(10), num_images_per_label)
}

with open('small-cifar-10-batches-py/test_batch', 'wb') as f:
    pickle.dump(_data, f)

with tarfile.open('small-cifar-10-python.tar.gz', 'w:gz') as tar:
    tar.add('small-cifar-10-batches-py')
