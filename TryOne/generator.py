# -*- coding: utf-8 -*-

import pandas as pd
import pathlib
import os
import pickle
from datetime import datetime, timedelta

from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from preprocessing.dataSet import DataSet
from classifier.CNN.Conf1DClassifier02 import Conf1DClassifier
from classifier.PredictionHistory import PredictionHistory
from misc.helper import calcCategories

_version = 0.5
_epoch = 100
weekDeltaProve = 1

useTrainData = True

# results/<base lvl>/<1th level>/<2nd lvl>

config = {
     'mainSymbol'             : 'EURUSD', # base lvl
#     'indicatorSymbols'       : ['EURGBP', 'GBPUSD', 'USDJPY', 'EURJPY'], # base lvl
     'indicatorSymbols'       : ['EURGBP', 'EURJPY'], # base lvl
#     'indicatorSymbols'       : ['EURGBP'], # base lvl

     'l2RegularizeVal'        : 0.001, # 'None' to dectivate # 1th lvl
#     'kernel_size'             : 9,
     'maxPooling'              : True,
#     'maxPoolingSize'          : 2,
#     'optimizer'               : 'adam',

#     'lookback_stepsize'      : 1, # 2nd lvl
#     'beginTrain'             : beginTrain, # 2nd lvl
#     'endTrain'               : endTrain, # 2nd lvl
#     'endTest'                : endTest, # 2nd lvl
#     'weekDeltaTrain'         : weekDeltaTrain, # 2nd lvl
#     'weekDeltaProve'         : weekDeltaProve, # 2nd lvl


     'lookback_batch'         : 24*60, # const
     'maxTimeDeltaAcceptance' : '1 days 1 hours', # const
     'forward_set_lengh'      : 60, # const
     'interpolateLimit'       : 60, # const
     'bounds'                 : { 'EURUSD' : 0.0010 }, # const
}


#l2RegVals_Options = [0.001, 0.01]
kernel_size_Options = [7, 9, 12, 15]
maxPoolingSize_Options = [2, 4]
optimizer_Options = ['adam', 'sgd', 'rmsprop', 'adagrad']

# permutation helper
# np.stack(np.meshgrid([1, 2, 3], [4, 5], [6, 7]), -1).reshape(-1, 3)
import numpy as np
firstLvlTestSet = np.stack(np.meshgrid(kernel_size_Options, optimizer_Options, 
                                     maxPoolingSize_Options), -1).reshape(-1, 3)


############
#   Main   #
############

##################################################################################################

def createDirs(base):
    tbDir = os.path.join(base, 'tensorboard')
    smDir = os.path.join(base, 'savedModels')

    pathlib.Path(tbDir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(smDir).mkdir(parents=True, exist_ok=True)
    
    return tbDir, smDir

def execute(config, X_train, y_train, X_test, y_test, cat_count, useTrainData):

    conf1D = Conf1DClassifier(config)

    classifier = conf1D.getClassifier(X_test.shape[2])

    tbDir, smDir = createDirs(config['resultPath'])
    
    with open(os.path.join(config['resultPath'], "config.pickle"), 'wb') as configFile:
        pickle.dump(config, configFile)

    tb_callback = TensorBoard(log_dir=tbDir)
    es_callback = EarlyStopping(monitor='val_loss', min_delta=0.001, verbose=1, patience=20)
    mc_callback = ModelCheckpoint(smDir + "/weights.{epoch:02d}-{acc:.4f}.hdf5", monitor='acc')
    ph_callback = PredictionHistory(X = X_test, Y = y_test, bound = 0.9)

    hist = classifier.fit(X_train, y_train,
               class_weight = cat_weights,
               epochs = _epoch, 
               validation_split=0.2,
               callbacks=[tb_callback, es_callback, mc_callback, ph_callback])
    
    resultSet = pd.concat([ph_callback.getPredHist(), pd.DataFrame(hist.history)], axis=1)
    resultSet.to_csv(path_or_buf = os.path.join(config['resultPath'], "predict.csv"), float_format = '%.3f', decimal = ',',
             columns = ['acc', 'loss', 'val_acc', 'val_loss', 'good', 'bad', 'unknown'])

    return hist.history

secndLvlTestSet = { 'date': [], 'trainWeeks': [], 'stepsize': []}
secndLvlTestSet = pd.DataFrame(data=secndLvlTestSet)
secndLvlTestSet = secndLvlTestSet.append({ 'date' : datetime(2017,12,10), 'trainWeeks': 2, 'stepsize': 1 }, ignore_index=True)


for i, row in secndLvlTestSet.iterrows():
    print(i, ":", row['date'], row['trainWeeks'], int(row['stepsize']))

    beginTrain = row['date'] - timedelta(weeks=row['trainWeeks'])
    endTest = row['date'] + timedelta(weeks=weekDeltaProve)
    
    config['endTrain'] = row['date']
    config['beginTrain'] = beginTrain
    config['endTest'] = endTest
    
    config['lookback_stepsize'] = int(row['stepsize'])
    
    dataSet = DataSet(config, useTrainData)
    trainSetRAW, testSetRAW = dataSet.getDataForSymbol(config['mainSymbol'])
    
    ## testing 
    
    trainSetRAW = trainSetRAW.resample('T').asfreq()
    
    from sklearn.preprocessing import MinMaxScaler
    
    # MACD
    trainSetRAW['26 ema'] = trainSetRAW["EURUSD"].ewm(span=26, min_periods=26).mean()
    trainSetRAW['12 ema'] = trainSetRAW["EURUSD"].ewm(span=12, min_periods=12).mean()
    trainSetRAW['MACD'] = (trainSetRAW['12 ema'] - trainSetRAW['26 ema'])
    
    sc = MinMaxScaler(feature_range = (-1, 1))
    trainSetRAW["MACD scaled"] = sc.fit_transform(trainSetRAW[['MACD']])

    # BB
    trainSetRAW['20 ma'] = trainSetRAW['EURUSD'].rolling(20).mean()
    trainSetRAW['20 sd'] = trainSetRAW['EURUSD'].rolling(20).std()
#    trainSetRAW['Upper Band'] = trainSetRAW['20 ma'] + (trainSetRAW['20 sd']*2)
#    trainSetRAW['Lower Band'] = trainSetRAW['20 ma'] - (trainSetRAW['20 sd']*2)

    trainSetRAW['Upper Band'] = trainSetRAW['20 sd']*2
    trainSetRAW['Lower Band'] = trainSetRAW['20 sd']*-2
    
    # RSI
    period = 14
    delta = trainSetRAW['EURUSD'].diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean( u[:period] )
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean( d[:period] )
    d = d.drop(d.index[:(period-1)])
    rs = u.ewm(com=period-1, adjust=False).mean() / d.ewm(com=period-1, adjust=False).mean()
    rsi = 100 - 100 / (1 + rs)
    trainSetRAW['RSI'] = rsi
    trainSetRAW["RSI scaled"] = (rsi - 50) / 100
    
    import matplotlib.pyplot as plt
    
    plt.plot(trainSetRAW["EURUSD"], 'g')
    plt.plot(trainSetRAW["26 ema"], 'b')
    plt.plot(trainSetRAW["12 ema"], 'r')
    plt.plot(trainSetRAW["MACD"], 'r')
    plt.plot(trainSetRAW["MACD scaled"], 'r')
    
    plt.plot(trainSetRAW["20 ma"], 'b')
    plt.plot(trainSetRAW["Upper Band"], 'r')
    plt.plot(trainSetRAW["Lower Band"], 'g')
    
    
    plt.plot(trainSetRAW["RSI scaled"], 'r')
    
    plt.show()
    
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    
    sc = MinMaxScaler(feature_range = (0.05, 1))
    
    a = np.array([2.,5.,6.,7.,12.])
    a = a-a.min()
    a_based = a.reshape(-1,1)
    
    b = np.array([14.,13.,12.,15.,17.])
    b = b-b.min()
    b_based = b.reshape(-1,1)

    c = np.array([34.,33.,30.,35.,37.])
    c = c-c.min()
    c_based = c.reshape(-1,1)
    
    a_plot = sc.fit_transform(a_based)
    b_plot = sc.transform(b_based)
    c_plot = sc.transform(c_based)
    
    plt.plot(a_plot, 'b')
    plt.plot(b_plot, 'r')
    plt.plot(c_plot, 'g')
    
    
    ## end

    for sym in config['indicatorSymbols']:
        _train, _test = dataSet.getDataForSymbol(sym)
    
        trainSetRAW = pd.concat([trainSetRAW, _train], axis=1, join_axes=[trainSetRAW.index])
        testSetRAW = pd.concat([testSetRAW, _test], axis=1, join_axes=[testSetRAW.index])

    trainSetRAW = trainSetRAW.dropna()
    testSetRAW = testSetRAW.dropna()

    X_train, y_train, X_test, y_test = dataSet.getXYArrays(trainSetRAW, testSetRAW)

    cat_count, cat_weights = calcCategories(y_train)
    
    sndLvlResultPath = "results/%s/%s_%s-%s" % (config['mainSymbol'],
              row['date'].strftime("%Y-%m-%d"), int(row['trainWeeks']), config['lookback_stepsize'])
    pathlib.Path(sndLvlResultPath).mkdir(parents=True, exist_ok=True)
    
    with open(os.path.join(sndLvlResultPath, "catCount.pickle"), 'wb') as catCountFile:
        pickle.dump(cat_count, catCountFile)
    
    for testSet in firstLvlTestSet:
#        config['l2RegularizeVal'] = float(testSet[3])
        config['kernel_size']     = int(testSet[0])
        config['maxPoolingSize']  = int(testSet[2])
        config['optimizer']       = testSet[1]

        config['resultPath'] = "%s/%s_%s_%s_%s" % (sndLvlResultPath, config['optimizer'],
               config['kernel_size'], config['maxPoolingSize'], config['l2RegularizeVal'])
        print("generating result for %s" % config['resultPath'])
        
        if os.path.exists(config['resultPath']):
            print ("Path %s already exists" % (config['resultPath']))
        else:
            hist = execute(config, X_train, y_train, X_test, y_test, cat_count, useTrainData)
            print(hist)
