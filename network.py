from __future__ import division, print_function, absolute_import

import numpy as np
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.conv import conv_2d, max_pool_2d, highway_conv_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

    
class CNN_1:
    """
    Simple version of CNN with no regularizer
    """
    network = None
    
    def __init__(self, learning_rate):
        
        network = input_data(shape=[None, 32, 32, 3])
        network = conv_2d(network, 32, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = conv_2d(network, 64, 3, activation='relu')
        network = conv_2d(network, 64, 3, activation='relu')
        network = max_pool_2d(network, 2)
        network = fully_connected(network, 512, activation='relu')
        network = dropout(network, 0.5)
        network = fully_connected(network, 10, activation='softmax')
        network = regression(network, optimizer='adam',
                             loss='categorical_crossentropy',
                             learning_rate=learning_rate)
        self.network = network
        
    def get_model(self):
        return self.network
    
    
class CNN_2:
    """
    Highway CNN
    """
    network = None
    
    def __init__(self, learning_rate):
        
         # Real-time data preprocessing
        img_prep = ImagePreprocessing()
        img_prep.add_featurewise_zero_center()
        img_prep.add_featurewise_stdnorm()


        # Convolutional network building
        network = input_data(shape=[None, 32, 32, 3],data_preprocessing=img_prep)
        #highway convolutions with pooling and dropout
        for i in range(3):
            for j in [3, 2, 1]: 
                network = highway_conv_2d(network, 16, j, activation='elu')
            network = max_pool_2d(network, 2)
            network = batch_normalization(network)

        network = fully_connected(network, 128, activation='elu')
        network = dropout(network, 0.5)
        network = fully_connected(network, 256, activation='elu')
        network = fully_connected(network, 10, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate= learning_rate,
                             loss='categorical_crossentropy')
        self.network = network
        
    def get_model(self):
        return self.network
    
class DNN:
    """
    DNN Network
    """
    network = None
    
    def __init__(self, learning_rate):
        input_layer = tflearn.input_data(shape=[None, 32,32,3])
        dense1 = tflearn.fully_connected(input_layer, 32, activation='relu')
        dropout1 = tflearn.dropout(dense1, 0.5)
        
        dense2 = tflearn.fully_connected(dropout1, 64, activation='relu')
        dropout2 = tflearn.dropout(dense2, 0.25)
        
        dense3 = tflearn.fully_connected(dropout2, 128, activation='relu')
        dropout3 = tflearn.dropout(dense3, 0.5)
        
        dense4 = tflearn.fully_connected(dropout3, 256, activation='relu')
        dropout4 = tflearn.dropout(dense4, 0.5)
     
        dense5 = tflearn.fully_connected(dropout4, 512, activation='relu')      
        
        softmax = tflearn.fully_connected(dense5, 10, activation='softmax')
        
        sgd = tflearn.SGD(learning_rate= learning_rate, lr_decay=0.95, decay_step=1000)
        network = tflearn.regression(softmax, optimizer= sgd, learning_rate=learning_rate,
                                     loss='categorical_crossentropy')
        self.network = network
        
    def get_model(self):
        return self.network