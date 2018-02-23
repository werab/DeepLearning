# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from keras.models import load_model

from predictor.Predictor import Predictor
from preprocessing.dataSet import DataSet

def getTestData(config, useTrainData):
    dataSet = DataSet(config, useTrainData)
    trainSetRAW, testSetRAW = dataSet.getDataForSymbol(config['mainSymbol'])

    for sym in config['indicatorSymbols']:
        _train, _test = dataSet.getDataForSymbol(sym)
    
        trainSetRAW = pd.concat([trainSetRAW, _train], axis=1, join_axes=[trainSetRAW.index])
        testSetRAW = pd.concat([testSetRAW, _test], axis=1, join_axes=[testSetRAW.index])

    trainSetRAW = trainSetRAW.dropna()
    testSetRAW = testSetRAW.dropna()

    _, _, X_test, y_test = dataSet.getXYArrays(trainSetRAW, testSetRAW)

    return X_test, y_test

classifier = load_model("./results/EURUSD/firstTry/2016-07-26_6-3/savedModels/weights.03-0.3057.hdf5")

config = pickle.load(open("./results/EURUSD/firstTry/2016-07-26_6-3/config.pickle", "rb"))

X_test, y_test = getTestData(config, False)

# Making the Confusion Matrix
predictor = Predictor(classifier, X_test, y_test, 0.9)

cm = predictor.getCM()

predictor.getStatistics()

# print("good: %.2f bad: %.2f unknown: %.2f" % (good/sumall, bad/sumall, unknown/sumall))


result_view = np.hstack((predictor.getYPred(), np.array(y_test).argmax(axis=1).reshape(-1,1)))

# View Plot
def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


for x in consecutive(predictor.getUpArray()):
    plt.axvspan(x[0]-1,x[-1], facecolor='g', alpha=0.2)

for x in consecutive(predictor.getDownArray()):
    plt.axvspan(x[0]-1,x[-1], facecolor='r', alpha=0.2)

X_plot = np.moveaxis(X_test, -1, 0)[0:1,:,1].flatten()
len(X_plot)

#plt.figure()
plt.plot(X_plot)
plt.show()