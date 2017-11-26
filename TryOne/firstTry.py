# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

load_saved_weights=False

# params
lookback_batch = 240 # 24*60
lookback_stepsize = 1

saved_weights = "firstTry_weights.h5"

forward_set_lengh = 60
bounds = { 'EURUSD' : 0.001 }

# categories
# 0: > value + bound                       --> buy 
# 1: < value - bound                       --> sell
# 2: < value + bound && > value - bound    --> nothing
# 3: > value + bound && < value - bound    --> buy && sell
def getCategory(value, np_forward_set):
    if (np_forward_set.max() > value + bounds['EURUSD']):
        if (np_forward_set.min() < value - bounds['EURUSD']):
            return [0,0,0,1]
        else:
            return [1,0,0,0]
    elif (np_forward_set.min() < value - bounds['EURUSD']):
        if (np_forward_set.max() > value + bounds['EURUSD']):
            return [0,0,0,1]
        else:
            return [0,1,0,0]
    return [0,0,1,0]

# Importing the training set
dataset_train = pd.read_csv('DAT_MT_EURUSD_M1_201710.csv', header=None)
dataset_train = dataset_train[:1000]
training_set = dataset_train.iloc[:, 2:3].values
training_set = training_set[:1000]

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []

for i in range(lookback_batch, len(training_set)-forward_set_lengh, lookback_stepsize):
    X_train.append(training_set_scaled[i-lookback_batch:i, 0])
    y_train.append(getCategory(training_set[i], np.array(training_set[i+1:i+forward_set_lengh])))
    
#X_train, y_train = np.array(X_train[:5000]), np.array(y_train[:5000])
X_train, y_train = np.array(X_train), np.array(y_train)

cat_count = [0,0,0,0]
for a in y_train:
    cat_count += a

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 1),
                   name = "LSTM_layer_one"))
#                   kernel_initializer = 'uniform', name = "LSTM_layer_one"))
#regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True,
                   name = "LSTM_layer_two"))
#                   kernel_initializer = 'uniform', name = "LSTM_layer_two"))
#regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True,
                   name = "LSTM_layer_three"))
#                   kernel_initializer = 'uniform', name = "LSTM_layer_three"))
#regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, name = "LSTM_layer_four"))
#regressor.add(LSTM(units = 50, kernel_initializer = 'uniform', name = "LSTM_layer_four"))
#regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 4))
#regressor.add(Dense(units = 4, kernel_initializer = 'uniform'))

# Compiling the RNN
regressor.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

if load_saved_weights:
    regressor.load_weights(saved_weights)
else:
    # Fitting the RNN to the Training set
    #regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
    regressor.fit(X_train, y_train, epochs = 100, batch_size = 10)
    regressor.save(saved_weights)

# todo: nochmal durchgehen !!!
dataset_test = pd.read_csv('DAT_MT_EURUSD_M1_201711.csv', header=None)
dataset_test = dataset_test[:1000]
real_forex = dataset_test.iloc[:, 2:3].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train[2], dataset_test[2]), axis = 0)
test_set = dataset_total[len(dataset_total) - len(dataset_test) - lookback_batch:].values
inputs = test_set.reshape(-1,1) # nochmal nachschauen im Lehrgang
inputs = sc.transform(inputs)
X_test = []
y_test = []
for i in range(lookback_batch, len(test_set)-forward_set_lengh):
    X_test.append(inputs[i-lookback_batch:i, 0])
    y_test.append(getCategory(test_set[i], np.array(test_set[i+1:i+forward_set_lengh])))
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_pred = regressor.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#upper = value + bounds['EURUSD']
#lower = value - bounds['EURUSD']
#plt.plot(np_forward_set, color = 'red')
#plt.hlines(np_forward_set.min(), 0, np_forward_set.size, color = 'blue')
#plt.hlines(np_forward_set.max(), 0, np_forward_set.size, color = 'green')
#plt.hlines(upper, 0, np_forward_set.size, color = 'yellow')
#plt.hlines(lower, 0, np_forward_set.size, color = 'yellow')
#plt.show()


#nparr = np.array(training_set[0:20])

#plt.plot(nparr, color = 'red')
#plt.hlines(nparr.min(), 0, nparr.size, color = 'blue')
#plt.hlines(nparr.max(), 0, nparr.size, color = 'green')
#plt.title("eine heiße suppe")
#plt.show()