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
        self.regVal = config['l2RegularizeVal']

    def getClassifier(self, dataSetCount, cat_count):
        # Initialising the CNN
        classifier = Sequential()
        
        # Step 1 - Convolution
        classifier.add(Conv1D(32, 9, input_shape = (self.lookback_batch, dataSetCount),
                              activation = 'relu',
                              dilation_rate = 1,
                              padding = 'causal',
                              kernel_regularizer = regularizers.l2(self.regVal)))#
                              #activity_regularizer = regularizers.l2(self.regVal)))
        # Step 2 - Pooling
        classifier.add(MaxPooling1D(pool_size = 2))
        
        # Adding a second convolutional layer
        classifier.add(Conv1D(32, 9, activation = 'relu',
                              dilation_rate = 2, padding = 'causal',
                              kernel_regularizer = regularizers.l2(self.regVal))),
                              #activity_regularizer = regularizers.l2(self.regVal)))
        classifier.add(MaxPooling1D(pool_size = 2))
        
        classifier.add(Conv1D(32, 9, activation = 'relu',
                              dilation_rate = 4, padding = 'causal',
                              kernel_regularizer = regularizers.l2(self.regVal))),
                              #activity_regularizer = regularizers.l2(self.regVal)))
        classifier.add(MaxPooling1D(pool_size = 2))
        
        # Step 3 - Flattening
        classifier.add(Flatten())
        
        # Step 4 - Full connection
        classifier.add(Dense(units = 128, activation = 'relu'))
        classifier.add(Dense(units = len(cat_count), activation = 'softmax'))
        
        # Compiling the CNN
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
        return classifier