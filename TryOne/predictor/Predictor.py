# -*- coding: utf-8 -*-

from sklearn.metrics import confusion_matrix
import numpy as np

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
        cm_cond = confusion_matrix(np.array(y_test_cond).argmax(axis=1), 
                                   y_pred_cond.argmax(axis=1), labels=[0,1,2])
        
        good = cm_cond[0][0] + cm_cond[1][1]
        bad = cm_cond[1][0] + cm_cond[0][1]
        unknown = cm_cond[0][2] + cm_cond[1][2]
        sumall = good + bad + unknown
        
        return { 'good'    : 0 if good    == 0 else good/sumall, 
                 'bad'     : 0 if bad     == 0 else bad/sumall, 
                 'unknown' : 0 if unknown == 0 else unknown/sumall }

    def getUpArray(self):
        return self.y_pred_i[np.where((self.y_pred_i[:,1] == 0))[0], :][:,0]
    
    def getDownArray(self):
        return self.y_pred_i[np.where((self.y_pred_i[:,1] == 1))[0], :][:,0]