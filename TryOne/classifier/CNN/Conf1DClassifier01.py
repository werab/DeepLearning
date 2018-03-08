# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
from keras import regularizers

class Conf1DClassifier():
    def __init__(self, config):
        self.lookback_batch = config['lookback_batch']
        
        # variable config
        self.kernel_size = config['kernel_size']
        self.regVal = config['l2RegularizeVal']
        self.maxPooling = config['maxPooling']
        self.maxPoolingSize = config['maxPoolingSize']
        self.optimizer = config['optimizer']

    def getClassifier(self, dataSetCount):
        # Initialising the CNN
        classifier = Sequential()
        
        # Step 1 - Convolution
        classifier.add(Conv1D(32, self.kernel_size, input_shape = (self.lookback_batch, dataSetCount),
                              activation = 'relu',
                              dilation_rate = 1,
                              padding = 'causal',
                              kernel_regularizer = regularizers.l2(self.regVal)))
        # Step 2 - Pooling
        if self.maxPooling:
            classifier.add(MaxPooling1D(pool_size = self.maxPoolingSize))
        
        # Adding a second convolutional layer
        classifier.add(Conv1D(32, self.kernel_size, activation = 'relu',
                              dilation_rate = 2, padding = 'causal',
                              kernel_regularizer = regularizers.l2(self.regVal)))

        if self.maxPooling:
            classifier.add(MaxPooling1D(pool_size = self.maxPoolingSize))
        
        classifier.add(Conv1D(32, self.kernel_size, activation = 'relu',
                              dilation_rate = 4, padding = 'causal',
                              kernel_regularizer = regularizers.l2(self.regVal))),

        if self.maxPooling:
            classifier.add(MaxPooling1D(pool_size = self.maxPoolingSize))
        
        # Step 3 - Flattening
        classifier.add(Flatten())
        
        # Step 4 - Full connection
        classifier.add(Dense(units = 128, activation = 'relu'))
        classifier.add(Dense(units = 3, activation = 'softmax'))
        
        # Compiling the CNN
        classifier.compile(optimizer = self.optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
        return classifier