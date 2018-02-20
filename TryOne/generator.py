# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

_version = 0.5
_epoch = 10


load_saved_weights=False
load_weights_file = "H5/v0.5_ep100_weekDeltaTrain48.h5"

fileregex = '*201801*'
weekDeltaTrain = 48
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

# results/<base lvl>/<1th level>/<2nd lvl>

save_weights = "v%s_ep%s_weekDeltaTrain%s.h5" % (_version, _epoch, weekDeltaTrain)

config = {
     'mainSymbol'             : 'EURUSD', # base lvl
     'indicatorSymbols'       : ['EURGBP', 'GBPUSD', 'USDJPY', 'EURJPY'], # base lvl

     'l2RegularizeVal'        : 0.005, # 'None' to dectivate # 1th lvl

     'lookback_stepsize'      : 1, # 2nd lvl
     'fileregex'              : fileregex, # 2nd lvl
     'beginTrain'             : beginTrain, # 2nd lvl
     'endTrain'               : endTrain, # 2nd lvl
     'endTest'                : endTest, # 2nd lvl
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


secndLvlTestSet = { 'date': [], 'trainWeeks': [], 'stepsize': [] }
a = pd.DataFrame(data=secndLvlTestSet)

a = a.append({ 'date' : datetime(2017,12,31), 'trainWeeks': 52, 'stepsize': 1 }, ignore_index=True)

for i, row in a.iterrows():
    print(row['date'], row['trainWeeks'], row['stepsize'])
    print("***")

resultPath = "results/%s/firstTry/%s_%s-%s" % (config['mainSymbol'], endTrain.strftime("%Y-%m-%d"), 
                weekDeltaTrain, config['lookback_stepsize'])

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



from classifier.CNN.Conf1DClassifier import Conf1DClassifier

conf1D = Conf1DClassifier(config)

classifier = conf1D.getClassifier(X_test.shape[2], cat_count)


from predictor.Predictor import Predictor

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
es_callback = EarlyStopping(monitor='acc', min_delta=0.005, verbose=1, patience=3)
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
