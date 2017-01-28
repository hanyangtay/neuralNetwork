"""
Loads MNIST image data
"""

import _pickle
import gzip

import numpy as np


def load_data():
    """
    Returns MNIST data as a tuple that comprises 50,000 training data entries,
    10,000 validation data entries and 10,000 test data entries.
    
    Each data entry is a tuple (x, y)
    x is a numpy ndarray of images.
    Each image is a numpy ndarray with 784 values for each pixel.
    
    y is a numpy ndarray of digits that corresponds to the image in x.
    """
    
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = _pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)
    
def load_data_nn():
    """
    Similar to load_data, but loads in a different format meant for NNs.
    Each column is a training example.

    training_inputs = [np.reshape(x, (784, 1)) for x in init_training[0]]
    training_results = [vectorised_result(y) for y in init_training[1]]
    training_data = list(zip(training_inputs, training_results))
    
    test_inputs = [np.reshape(x, (784, 1)) for x in init_test[0]]
    test_results = [vectorised_result(y) for y in init_test[1]]
    test_data = list(zip(test_inputs, test_results))
    
    
    """
    
    init_training, init_validation, init_test = load_data()
    
    training_inputs = [np.reshape(x, (784, 1)) for x in init_training[0]]
    training_results = [vectorised_result(y) for y in init_training[1]]
    training_data = list(zip(training_inputs, training_results))
    
    validation_X = np.hstack((np.reshape(x, (784, 1)) for x in init_validation[0]))
    validation_Y = np.hstack((vectorised_result(y) for y in init_validation[1]))
    validation_data = [validation_X, validation_Y]
    
    test_X = np.hstack((np.reshape(x, (784, 1)) for x in init_test[0]))
    test_Y = np.hstack((y for y in init_test[1]))
    test_data = [test_X, test_Y]
  
    return (training_data, validation_data, test_data)
    
def vectorised_result(y):
    result = np.zeros((10, 1))
    result[y] = 1.0
    return result
    
    
    