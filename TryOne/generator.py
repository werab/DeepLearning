# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import os
from datetime import datetime, timedelta

from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from preprocessing.dataSet import DataSet
from classifier.CNN.Conf1DClassifier import Conf1DClassifier
from predictor.Predictor import Predictor
from classifier.PredictionHistory import PredictionHistory

_version = 0.5
_epoch = 3
weekDeltaProve = 2


load_saved_weights=False
load_weights_file = "H5/v0.5_ep100_weekDeltaTrain48.h5"

#weekDeltaTrain = 48

#endTrain = datetime(2018,1,14) # year, month, day
useTrainData = True

# results/<base lvl>/<1th level>/<2nd lvl>

config = {
     'mainSymbol'             : 'EURUSD', # base lvl
#     'indicatorSymbols'       : ['EURGBP', 'GBPUSD', 'USDJPY', 'EURJPY'], # base lvl
     'indicatorSymbols'       : ['EURGBP'], # base lvl

     'l2RegularizeVal'        : 0.005, # 'None' to dectivate # 1th lvl

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


# permutation helper
# np.stack(np.meshgrid([1, 2, 3], [4, 5], [6, 7]), -1).reshape(-1, 3)


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

def createDirs(base):
    tbDir = os.path.join(base, 'tensorboard')
    smDir = os.path.join(base, 'savedModels')

    pathlib.Path(tbDir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(smDir).mkdir(parents=True, exist_ok=True)
    
    return tbDir, smDir

def execute(config, useTrainData):
    dataSet = DataSet(config, useTrainData)
    trainSetRAW, testSetRAW = dataSet.getDataForSymbol(config['mainSymbol'])

    for sym in config['indicatorSymbols']:
        _train, _test = dataSet.getDataForSymbol(sym)
    
        trainSetRAW = pd.concat([trainSetRAW, _train], axis=1, join_axes=[trainSetRAW.index])
        testSetRAW = pd.concat([testSetRAW, _test], axis=1, join_axes=[testSetRAW.index])

    trainSetRAW = trainSetRAW.dropna()
    testSetRAW = testSetRAW.dropna()

    X_train, y_train, X_test, y_test = dataSet.getXYArrays(trainSetRAW, testSetRAW)

    cat_count, cat_weights = calcCategories(y_train)

    conf1D = Conf1DClassifier(config)

    classifier = conf1D.getClassifier(X_test.shape[2], cat_count)

    tbDir, smDir = createDirs(config['resultPath'])

    tb_callback = TensorBoard(log_dir=tbDir)
    es_callback = EarlyStopping(monitor='acc', min_delta=0.005, verbose=1, patience=3)
    mc_callback = ModelCheckpoint(smDir + "/weights.{epoch:02d}-{acc:.4f}.hdf5", monitor='acc',save_best_only=True)
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
secndLvlTestSet = secndLvlTestSet.append({ 'date' : datetime(2016,7,26), 'trainWeeks': 2,   'stepsize': 1 }, ignore_index=True)
secndLvlTestSet = secndLvlTestSet.append({ 'date' : datetime(2016,7,26), 'trainWeeks': 2*2, 'stepsize': 2 }, ignore_index=True)
secndLvlTestSet = secndLvlTestSet.append({ 'date' : datetime(2016,7,26), 'trainWeeks': 2*3, 'stepsize': 3 }, ignore_index=True)

for i, row in secndLvlTestSet.iterrows():
    print(i, ":", row['date'], row['trainWeeks'], int(row['stepsize']))

    beginTrain = row['date'] - timedelta(weeks=row['trainWeeks'])
    endTest = row['date'] + timedelta(weeks=weekDeltaProve)
    
    config['endTrain'] = row['date']
    config['beginTrain'] = beginTrain
    config['endTest'] = endTest
    
    config['lookback_stepsize'] = int(row['stepsize'])

    config['resultPath'] = "results/%s/firstTry/%s_%s-%s" % (config['mainSymbol'], row['date'].strftime("%Y-%m-%d"), 
                row['trainWeeks'], config['lookback_stepsize'])
    
    hist = execute(config, useTrainData)
    print(hist)
