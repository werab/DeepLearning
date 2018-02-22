# -*- coding: utf-8 -*-

import pandas as pd
from keras.callbacks import Callback
from predictor.Predictor import Predictor

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