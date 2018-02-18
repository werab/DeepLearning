# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix


_version = 0.5
_epoch = 10

# todos:

# home / weekend
## class split
### classifier
### predictor
## result loggin
### to csv / pickle
### top acc / good 

# param optimisation
## automatic testing
## define stuff before testing
##    - https://keras.io/callbacks/
##      - propper result logging
##      - earlyStopping -!!-
##          - param optimization
##      - tensorboard -!!-
##          - param optimization
##    - config logging
##    - weights file -!!-
##    - learning log
##    - prediction result logging (good/bad/unknown)
##    - general directory structure

# results/<base lvl>/<1th level>/<2nd lvl>
# - csv / weights / tensorboard

# visual
## https://keras.io/visualization/

# Conv2D specifications (cumute / work)
## convolution window sizing

# classifier.fit optimisation

# general optimisation techniques
## https://cambridgespark.com/content/tutorials/neural-networks-tuning-techniques/index.html

# initialisation weights
## https://arxiv.org/pdf/1703.04691.pdf
## https://stackoverflow.com/questions/46798708/keras-how-to-view-initialized-weights-i-e-before-training

# leaning rate scheduling
# google search: keras learning rate decay
## https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
## https://github.com/keras-team/keras/issues/898

# set categories with from sklearn.preprocessing import LabelEncoder, OneHotEncoder library
## help https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/

# course from marco
## https://www.coursera.org/learn/machine-learning
# go article
## https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ


load_saved_weights=False
load_weights_file = "H5/v0.5_ep100_weekDeltaTrain48.h5"

# params
#lookback_batch = 24*60
#lookback_stepsize = 1
#maxTimeDeltaAcceptance = '1 days 1 hours'

fileregex = '*201801*'
weekDeltaTrain = 2*1
weekDeltaProve = 2
endTrain = datetime(2018,1,14) # year, month, day
useTrainData = True

#fileregex = '*2017*'
#weekDeltaTrain = 48*1
#weekDeltaProve = 4
#endTrain = datetime(2017,12,3) # year, month, day
#useTrainData = False

beginTrain = endTrain - timedelta(weeks=weekDeltaTrain)
endTest = endTrain + timedelta(weeks=weekDeltaProve)

save_weights = "v%s_ep%s_weekDeltaTrain%s.h5" % (_version, _epoch, weekDeltaTrain)

#mainSymbol = 'EURUSD'
#indicatorSymbols = ['EURGBP', 'GBPUSD', 'USDJPY', 'EURJPY']
#indicatorSymbols = ['EURGBP', 'EURJPY']
#indicatorSymbols = ['EURJPY']
#interpolateLimit = 60

#forward_set_lengh = 60
#bounds = { 'EURUSD' : 0.0010 }

config = {
     'mainSymbol'             : 'EURUSD', # base lvl
     'indicatorSymbols'       : ['EURGBP', 'GBPUSD', 'USDJPY', 'EURJPY'], # base lvl

     'l2RegularizeVal'        : 0.005, # 'None' to dectivate # 1th lvl

     'lookback_stepsize'      : 1, # 2nd lvl
     'fileregex'              : fileregex, # 2nd lvl
     'beginTrain'             : beginTrain, # 2nd lvl
     'endTrain'               : endTrain, # 2nd lvl
     'endTest'                : endTest, # 2nd lvl
     'weekDeltaTrain'         : weekDeltaTrain, # 2nd lvl
     'weekDeltaProve'         : weekDeltaProve, # 2nd lvl

     'lookback_batch'         : 24*60, # const
     'maxTimeDeltaAcceptance' : '1 days 1 hours', # const
     'forward_set_lengh'      : 60, # const
     'interpolateLimit'       : 60, # const
     'bounds'                 : { 'EURUSD' : 0.0010 }, # const
}


from preprocessing.dataSet import DataSet

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

dataSet = DataSet(config, useTrainData)

trainSetRAW, testSetRAW = dataSet.getDataForSymbol(config['mainSymbol'], fileregex)

for sym in config['indicatorSymbols']:
    _train, _test = dataSet.getDataForSymbol(sym, fileregex)

    trainSetRAW = pd.concat([trainSetRAW, _train], axis=1, join_axes=[trainSetRAW.index])
    testSetRAW = pd.concat([testSetRAW, _test], axis=1, join_axes=[testSetRAW.index])

trainSetRAW = trainSetRAW.dropna()
testSetRAW = testSetRAW.dropna()

X_train, y_train, X_test, y_test = dataSet.getXYArrays(trainSetRAW, testSetRAW)

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
from keras import regularizers
from keras import initializers

class Conf1DClassifier():
    def __init__(self, config, dataSetCount):
        self.lookback_batch = config['lookback_batch']
        self.regVal = config['l2RegularizeVal']
        
        self.dataSetCount = dataSetCount

    def getClassifier(self):
        # Initialising the CNN
        classifier = Sequential()
        
        # Step 1 - Convolution
        classifier.add(Conv1D(32, 9, input_shape = (self.lookback_batch, self.dataSetCount),
                              activation = 'relu',
                              dilation_rate = 1,
                              padding = 'causal',
                              kernel_regularizer = regularizers.l2(self.regVal),
                              activity_regularizer = regularizers.l2(self.regVal)))
        # Step 2 - Pooling
        classifier.add(MaxPooling1D(pool_size = 2))
        
        # Adding a second convolutional layer
        classifier.add(Conv1D(32, 9, activation = 'relu',
                              dilation_rate = 2, padding = 'causal',
                              kernel_regularizer = regularizers.l2(self.regVal),
                              activity_regularizer = regularizers.l2(self.regVal)))
        classifier.add(MaxPooling1D(pool_size = 2))
        
        classifier.add(Conv1D(32, 9, activation = 'relu',
                              dilation_rate = 4, padding = 'causal',
                              kernel_regularizer = regularizers.l2(self.regVal),
                              activity_regularizer = regularizers.l2(self.regVal)))
        classifier.add(MaxPooling1D(pool_size = 2))
        
        # Step 3 - Flattening
        classifier.add(Flatten())
        
        # Step 4 - Full connection
        classifier.add(Dense(units = 128, activation = 'relu'))
        classifier.add(Dense(units = len(cat_count), activation = 'softmax'))
        
        # Compiling the CNN
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
        return classifier

conf1D = Conf1DClassifier(config, X_test.shape[2])

classifier = conf1D.getClassifier()






class Predictor():
    def __init__(self, cf, X_test, y_test, limit):
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = cf.predict(X_test)
        self.y_pred_i = np.argwhere(self.y_pred > limit)
    
    def getYPred(self):
        return self.y_pred
    
    def getCM(self):
        return confusion_matrix(np.array(self.y_test).argmax(axis=1), 
                                self.y_pred.argmax(axis=1))

    def getStatistics(self):
        if len(self.y_pred_i) == 0:
            return { 'good' : 0.0, 'bad' : 0.0, 'unknown' : 1.0 }
        
        y_pred_cond = np.take(self.y_pred, np.moveaxis(self.y_pred_i, 0, -1)[0], axis=0)
        y_test_cond = np.take(self.y_test, np.moveaxis(self.y_pred_i, 0, -1)[0], axis=0)
        cm_cond = confusion_matrix(np.array(y_test_cond).argmax(axis=1), y_pred_cond.argmax(axis=1))
        
        good = cm_cond[0][0] + cm_cond[1][1]
        bad = cm_cond[1][0] + cm_cond[0][1]
        unknown = cm_cond[0][2] + cm_cond[1][2]
        sumall = good + bad + unknown
        
        return { 'good' : good/sumall, 'bad' : bad/sumall, 'unknown' : unknown/sumall }

    def getUpArray(self):
        return self.y_pred_i[np.where((self.y_pred_i[:,1] == 0))[0], :][:,0]
    
    def getDownArray(self):
        return self.y_pred_i[np.where((self.y_pred_i[:,1] == 1))[0], :][:,0]

from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback

class PredictionHistory(Callback):
    def __init__(self, X, Y, bound):
        d = { 'good': [], 'bad': [], 'unknown': [], 'ypred': [] }
        self.predhis = pd.DataFrame(data=d)
        self.X_test = X
        self.y_test = Y
        self.bound = bound
        
    def on_epoch_end(self, epoch, logs={}):
        p = Predictor(self.model, self.X_test, self.y_test, self.bound)
        stats = p.getStatistics()
        ypred = { "ypred" : p.getYPred()}
        self.predhis = self.predhis.append({ **stats, **ypred}, ignore_index=True)
    def getPredHist(self):
        return self.predhis

tb_callback = TensorBoard(log_dir='./logs')
es_callback = EarlyStopping(monitor='val_loss', min_delta=0.005, verbose=1, patience=3)
mc_callback = ModelCheckpoint("./logs/weights.{epoch:02d}-{acc:.4f}.hdf5", monitor='acc',save_best_only=True)
ph_callback = PredictionHistory(X = X_test, Y = y_test, bound = 0.9)


if load_saved_weights:
    classifier.load_weights(load_weights_file)
else:
    hist = classifier.fit(X_train, y_train,
                   class_weight = cat_weights,
                   epochs = _epoch, 
                   validation_split=0.2,
                   callbacks=[tb_callback, es_callback, mc_callback, ph_callback])
    print(hist.history)
    classifier.save(save_weights)


df = pd.concat([ph_callback.getPredHist(), pd.DataFrame(hist.history)], axis=1)


df.to_csv(path_or_buf = './logs/test.csv', float_format = '%.3f', decimal = ',',
         columns = ['acc', 'loss', 'val_acc', 'val_loss', 'good', 'bad', 'unknown'])

   

# Making the Confusion Matrix
predictor = Predictor(classifier, X_test, y_test, 0.9)

cm = predictor.getCM()

predictor.getStatistics()

# print("good: %.2f bad: %.2f unknown: %.2f" % (good/sumall, bad/sumall, unknown/sumall))


result_view = np.hstack((predictor.getYPred(), np.array(y_test).argmax(axis=1).reshape(-1,1)))

# View Plot
def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

if load_saved_weights:
    for x in consecutive(predictor.getUpArray()):
        plt.axvspan(x[0]-1,x[-1], facecolor='g', alpha=0.2)
    
    for x in consecutive(predictor.getDownArray()):
        plt.axvspan(x[0]-1,x[-1], facecolor='r', alpha=0.2)
    
    X_plot = np.moveaxis(X_test, -1, 0)[0:1,:,1].flatten()
    len(X_plot)
    
    #plt.figure()
    plt.plot(X_plot)
    plt.show()

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
