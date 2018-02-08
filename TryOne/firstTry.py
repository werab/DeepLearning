# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import glob
from sklearn.preprocessing import MinMaxScaler
import sys

_version = 0.5
_epoch = 100

# todos:

# Conv1D --> Conv2D
## load multible rows
## arrange X_train with multible datasets

# https://keras.io/visualization/

# think abount new prediction accurancy (home)
# visual

# Conv2D specifications (cumute / work)
## convolution window sizing

# param optimisation
## automatic testing
## saved weights to paramnamed.save.file

# classifier.fit optimisation

# initialisation weights
## https://arxiv.org/pdf/1703.04691.pdf
# set categories with from sklearn.preprocessing import LabelEncoder, OneHotEncoder library
## help https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/

load_saved_weights=False
load_weights_file = "H5/v0.3_200epoch_lookback5_year2015.h5"

# params
lookback_batch = 24*60
lookback_stepsize = 1
maxTimeDeltaAcceptance = '1 days 1 hours'

#fileregex = '*201801*'
#weekDeltaTrain = 2*1
#weekDeltaProve = 2
#endTrain = datetime(2018,1,14) # year, month, day

fileregex = '*2017*'
weekDeltaTrain = 48*1
weekDeltaProve = 4
endTrain = datetime(2017,12,3) # year, month, day

beginTrain = endTrain - timedelta(weeks=weekDeltaTrain)
endTest = endTrain + timedelta(weeks=weekDeltaProve)

save_weights = "v%s_ep%s_weekDeltaTrain%s.h5" % (_version, _epoch, weekDeltaTrain)

mainSymbol = 'EURUSD'
indicatorSymbols = ['EURGBP', 'GBPUSD', 'USDJPY', 'EURJPY']
#indicatorSymbols = ['EURGBP', 'EURJPY']
#indicatorSymbols = ['EURJPY']
interpolateLimit = 60

forward_set_lengh = 60
bounds = { 'EURUSD' : 0.0010 }

# const
dateparse = lambda x: pd.datetime.strptime(x, '%Y.%m.%d %H:%M')

# categories
# 0: > value + bound                       --> buy 
# 1: < value - bound                       --> sell
# 2: < value + bound && > value - bound    --> nothing
def getCategory(value, np_forward_set):
    if (np_forward_set.max() > value + bounds[mainSymbol]):
        if (np_forward_set.min() < value - bounds[mainSymbol]):
            # both but direction first
            if (np_forward_set.argmin() < np_forward_set.argmax()):
                return [0,1,0]
            else:
                return [1,0,0]
        else:
            return [1,0,0]
    elif (np_forward_set.min() < value - bounds[mainSymbol]):
        if (np_forward_set.max() > value + bounds[mainSymbol]):
            # both but direction first
            if (np_forward_set.argmin() < np_forward_set.argmax()):
                return [0,1,0]
            else:
                return [1,0,0]
        else:
            return [0,1,0]
    return [0,0,1]

def getStructuredData(dataset, orignal_set, scaled_set, symbol):
    x = []
    y = []

    # idx of new week beginnings
    week_change_idx = np.array(dataset.reset_index()['datetime'].diff() > pd.Timedelta(maxTimeDeltaAcceptance)).nonzero()
    week_change_idx = np.append(week_change_idx, len(orignal_set))
    
    week_start_idx = 0
    for week_end_idx in np.nditer(week_change_idx):
    #    print("from: ", week_start_idx, " to: ", week_end_idx, " diff: ", week_end_idx-week_start_idx)
    #    print("next range from: ", week_start_idx+lookback_batch, " to: ", week_end_idx-forward_set_lengh)
        range_from = week_start_idx + lookback_batch
        range_to = week_end_idx - forward_set_lengh
        if range_from >= range_to:
            continue
        for i in range(range_from, range_to, lookback_stepsize):
            x.append(scaled_set[i-lookback_batch:i, 0])
            if symbol == mainSymbol:
                y.append(getCategory(orignal_set[i], np.array(orignal_set[i+1:i+forward_set_lengh])))
        week_start_idx = week_end_idx
    
    return x, y

# symbol matches directory
# file used as filter (for testing)
def loadSymbolCSV(symbol, file='*'):
    df = None
    for file in glob.glob("%s/%s" % (symbol, file)):
        print("Load: ", file)
        next_df = pd.read_csv(file, header=None, index_col = 'datetime',
                         parse_dates={'datetime': [0, 1]}, 
                         date_parser=dateparse)
        df = pd.concat([df, next_df])
    return df

# Importing training/test set
#def getSymbolData(symbol, fileregex):
#    # parse 0/1 column to datetime column
#    dataset_raw = None
#    try:
#        dataset_raw = loadSymbolCSV(symbol, fileregex).sort_index()
#    except:
#        print("missing data for symbol %s for regex '%s' 0 rows found." % (symbol, fileregex))
#        raise
#    
#    dataset_inter = dataset_raw.resample('1T').asfreq().interpolate(method='quadratic', limit=interpolateLimit).dropna().reset_index()
#
#    dataset_train = dataset_inter[(dataset_inter['datetime'] > beginTrain) & (dataset_inter['datetime'] < endTrain)]
#    dataset_train = dataset_train.reset_index(drop=True)
#    training_set = dataset_train.iloc[:, 1:2].values
#    
#    dataset_test = dataset_inter[(dataset_inter['datetime'] > endTrain) & (dataset_inter['datetime'] < endTest)]
#    dataset_test = dataset_test.reset_index(drop=True)
#    test_set = dataset_test.iloc[:, 1:2].values
#    
#    # Feature Scaling
#    sc = MinMaxScaler(feature_range = (0, 1))
#    training_set_scaled = sc.fit_transform(training_set)
#    
#    x_arr_train, y_arr_train = getStructuredData(dataset_train, training_set, training_set_scaled, symbol)
#    x_arr_test, y_arr_test = getStructuredData(dataset_test, test_set, sc.transform(test_set), symbol)
#    
#    return x_arr_train, y_arr_train, x_arr_test, y_arr_test

def getXYArrays(datasetTrain, datasetTest):
    sc = MinMaxScaler(feature_range = (0, 1))
    
    ## Main Symbol ##
    symArrTrainMain = datasetTrain[mainSymbol]
    training_set_main = np.array([symArrTrainMain.values]).reshape(-1,1)
    training_set_scaled_main = sc.fit_transform(training_set_main)
    
    symArrTestMain = datasetTest[mainSymbol]
    test_set_main = np.array([symArrTestMain.values]).reshape(-1,1)
    test_set_scaled_main = sc.transform(test_set_main)
    
    x_arr_train_main, y_arr_train_main = getStructuredData(
            symArrTrainMain, training_set_main, training_set_scaled_main, mainSymbol)
    x_arr_test_main, y_arr_test_main = getStructuredData(
            symArrTestMain, test_set_main, test_set_scaled_main, mainSymbol)
    
    y_train = np.array(y_arr_train_main)
    y_test = np.array(y_arr_test_main)
    
    X_train = [x_arr_train_main]
    X_test = [x_arr_test_main]
    
    # other indicator symbols
    for symbol in indicatorSymbols:
        print(datasetTrain[symbol])
        symArrTrain = datasetTrain[symbol]
        training_set = np.array([symArrTrain.values]).reshape(-1,1)
        training_set_scaled = sc.fit_transform(training_set)
        
        symArrTest = datasetTest[symbol]
        test_set = np.array([symArrTest.values]).reshape(-1,1)
        test_set_scaled = sc.transform(test_set)
    
        x_arr_train, y_arr_train = getStructuredData(
                symArrTrain, training_set, training_set_scaled, symbol)
        x_arr_test, y_arr_test = getStructuredData(
                symArrTest, test_set, test_set_scaled, symbol)

        X_train.append(x_arr_train)
        X_test.append(x_arr_test)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    # Reshaping
    X_train = np.moveaxis(X_train, 0, -1)
    X_test = np.moveaxis(X_test, 0, -1)

    return X_train, y_train, X_test, y_test

def getDataForSymbol(symbol, fileregex):
    # parse 0/1 column to datetime column
    dataset_raw = None
    try:
        dataset_raw = loadSymbolCSV(symbol, fileregex).sort_index()
    except:
        print("missing data for symbol %s for regex '%s' 0 rows found." % (symbol, fileregex))
        raise
    
    dataset_inter = dataset_raw.resample('1T').asfreq().interpolate(method='quadratic', limit=interpolateLimit).dropna()

    dataset_train = dataset_inter[(dataset_inter.index > beginTrain) & (dataset_inter.index < endTrain)]
    dataset_train = dataset_train.iloc[:, 0:1]
    dataset_train = dataset_train.rename(columns = {2: symbol})
    
    dataset_test = dataset_inter[(dataset_inter.index > endTrain) & (dataset_inter.index < endTest)]
    dataset_test = dataset_test.iloc[:, 0:1]
    dataset_test = dataset_test.rename(columns = {2: symbol})

    return dataset_train, dataset_test

def calcCategories(y_train):
    cat_count = [0,0,0]
    for a in y_train:
        cat_count += a
    
    # equal class distribution
    cat_weights = {}
    for idx, cat_class in enumerate(cat_count):
        cat_weights[idx] = round(np.max(cat_count)/cat_class)
        
    return cat_count, cat_weights

############
#   Main   #
############

##################################################################################################
# TEST

trainSetRAW, testSetRAW = getDataForSymbol(mainSymbol, fileregex)

for sym in indicatorSymbols:
    _train, _test = getDataForSymbol(sym, fileregex)

    trainSetRAW = pd.concat([trainSetRAW, _train], axis=1, join_axes=[trainSetRAW.index])
    testSetRAW = pd.concat([testSetRAW, _test], axis=1, join_axes=[testSetRAW.index])

trainSetRAW = trainSetRAW.dropna()
testSetRAW = testSetRAW.dropna()

X_train, y_train, X_test, y_test = getXYArrays(trainSetRAW, testSetRAW)

cat_count, cat_weights = calcCategories(y_train)

#mainS = trainSetRAW[mainSymbol]
#mainSt = np.array([mainS.values]).reshape(-1,1)
#sc = MinMaxScaler(feature_range = (0, 1))
#training_set_scaled = sc.fit_transform(mainSt)
#
#
#d = mainS.iloc[:, 1:2].values
#mainS.diff()
#
#mainSt = [mainS.values]
#
#mainSt = np.moveaxis(mainS, 0, -1)
#
#week_change_idx = np.array(mainS.reset_index()['datetime'].diff() > pd.Timedelta(maxTimeDeltaAcceptance)).nonzero()
#week_change_idx = np.append(week_change_idx, len(mainSt))

##################################################################################################

#x_arr_main, y_arr_main, x_arr_test, y_arr_test = getSymbolData(mainSymbol, fileregex)
#
#y_train = np.array(y_arr_main)
#cat_count, cat_weights = calcCategories(y_train)
#
#X_train = [x_arr_main]
#X_test = [x_arr_test]
#
#print("Symbol: %s Size: %i" % (mainSymbol, len(x_arr_main)))
#
##scale_hash = { mainSymbol : sc_main }
##test_set_hash = { mainSymbol : test_set_main }
#
#for sym in indicatorSymbols:
#    print(sym)
#    x_arr_sym_train, _, x_arr_sym_test, _ = getSymbolData(sym, fileregex)
#    print("Symbol: %s Size: %i" % (sym, len(x_arr_sym_train)))
#    X_train.append(x_arr_sym_train)
#    X_test.append(x_arr_sym_test)
#    
#X_train = np.array(X_train)
#X_test = np.array(X_test)
#
## Reshaping
#X_train = np.moveaxis(X_train, 0, -1)
#X_test = np.moveaxis(X_test, 0, -1)

# Reshaping
#X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#a = np.array([[1.,1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09],
#              [1.1,1.11,1.12,1.13,1.14,1.15,1.16,1.17,1.18,1.19]])
#b = np.array([[2.,2.01,2.02,2.03,2.04,2.05,2.06,2.07,2.08,2.09],
#              [2.1,2.11,2.12,2.13,2.14,2.15,2.16,2.17,2.18,2.19]])
#c = np.array([[3.,3.01,3.02,3.03,3.04,3.05,3.06,3.07,3.08,3.09],
#              [3.1,3.11,3.12,3.13,3.14,3.15,3.16,3.17,3.18,3.19]])
#
#x = []
#x.append(a)
#x.append(b)
#x.append(c)
#
#d = np.array(x)
#d.shape
#np.moveaxis(d, 0, -1).shape

# calculate weights
# Creating your class_weight dictionary:
# 1. determine the ratio of reference_class/other_class. If you choose class_0 as your reference, you'll have (1000/1000, 1000/500, 1000/100) = (1,2,10)
# 2. map the class label to the ratio: class_weight={0:1, 1:2, 2:10}

# get max class index
# calc maxclass_val / other classes without rest

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
from keras import initializers

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv1D(32, 9, input_shape = (lookback_batch, X_train.shape[2]), activation = 'relu',
                      dilation_rate = 1, padding = 'causal'))
# Step 2 - Pooling
classifier.add(MaxPooling1D(pool_size = 2))

# Adding a second convolutional layer
classifier.add(Conv1D(32, 9, activation = 'relu',
                      dilation_rate = 2, padding = 'causal'))
classifier.add(MaxPooling1D(pool_size = 2))

classifier.add(Conv1D(32, 9, activation = 'relu',
                      dilation_rate = 4, padding = 'causal'))
classifier.add(MaxPooling1D(pool_size = 2))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = len(cat_count), activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


if load_saved_weights:
    classifier.load_weights(load_weights_file)
else:
    classifier.fit(X_train, y_train,
                   class_weight = cat_weights,
                   epochs = _epoch, 
                   validation_split=0.2)
    classifier.save(save_weights)

sys.exit(0)

# Importing the Keras libraries and packages
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
#from keras.layers import Dropout

# Initialising the RNN
#regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
#regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 1),
#                   name = "LSTM_layer_one"))
#                   kernel_initializer = 'uniform', name = "LSTM_layer_one"))
#regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
#regressor.add(LSTM(units = 100, return_sequences = True,
#                   name = "LSTM_layer_two"))
#                   kernel_initializer = 'uniform', name = "LSTM_layer_two"))
#regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
#regressor.add(LSTM(units = 50, return_sequences = True,
#                   name = "LSTM_layer_three"))
#                   kernel_initializer = 'uniform', name = "LSTM_layer_three"))
#regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
#regressor.add(LSTM(units = 50, name = "LSTM_layer_four"))
#regressor.add(LSTM(units = 50, kernel_initializer = 'uniform', name = "LSTM_layer_four"))
#regressor.add(Dropout(0.2))

# Adding the output layer
#regressor.add(Dense(units = 4))
#regressor.add(Dense(units = 4, kernel_initializer = 'uniform'))

# Compiling the RNN
#regressor.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#if load_saved_weights:
#    regressor.load_weights(save_weights)
#else:
    # Fitting the RNN to the Training set
    #regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
#    regressor.fit(X_train, y_train, epochs = 100, batch_size = 10)
#    regressor.save(save_weights)

# todo: nochmal durchgehen !!!
#dataset_test = pd.read_csv('DAT_MT_EURUSD_M1_201711.csv', header=None)
#dataset_test = dataset_test
#real_forex = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
#dataset_total = pd.concat((dataset_train[2], dataset_test[2]), axis = 0)
#test_set = dataset_total[len(dataset_total) - len(dataset_test) - lookback_batch:].values

    
#test_set_scaled = test_set.reshape(-1,1) # nochmal nachschauen im Lehrgang
#test_set_scaled = sc.transform(test_set_scaled)

#X_test = []
#y_test = []
#for i in range(lookback_batch, len(test_set)-forward_set_lengh):
#    X_test.append(test_set_scaled[i-lookback_batch:i, 0])
#    y_test.append(getCategory(test_set[i], np.array(test_set[i+1:i+forward_set_lengh])))
#    

#X_test, y_test = getStructuredData(dataset_test, test_set, test_set_scaled)

#X_test = np.array(X_test)
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(np.array(y_test).argmax(axis=1), y_pred.argmax(axis=1))

result_view = np.hstack((y_pred, np.array(y_test).argmax(axis=1).reshape(-1,1)))


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
