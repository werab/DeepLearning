# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import glob
#from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

class DataSet():
    def __init__(self, config, getTrainData):
        self.mainSymbol             = config['mainSymbol']
        self.indicatorSymbols       = config['indicatorSymbols']
        self.lookback_batch         = config['lookback_batch']
        self.lookback_stepsize      = config['lookback_stepsize']
        self.maxTimeDeltaAcceptance = config['maxTimeDeltaAcceptance']
        self.interpolateLimit       = config['interpolateLimit']
        self.forward_set_lengh      = config['forward_set_lengh']
        self.bounds                 = config['bounds']
        self.beginTrain             = config['beginTrain']
        self.endTrain               = config['endTrain']
        self.endTest                = config['endTest']
        
        self.getTrainData = getTrainData

        self.dateparse = lambda x: pd.datetime.strptime(x, '%Y.%m.%d %H:%M')

    # categories
    # 0: > value + bound                       --> buy 
    # 1: < value - bound                       --> sell
    # 2: < value + bound && > value - bound    --> nothing
    def getCategory(self, value, np_forward_set):
        if (np_forward_set.max() > value + self.bounds[self.mainSymbol]):
            if (np_forward_set.min() < value - self.bounds[self.mainSymbol]):
                # both but direction first
                if (np_forward_set.argmin() < np_forward_set.argmax()):
                    return [0,1,0]
                else:
                    return [1,0,0]
            else:
                return [1,0,0]
        elif (np_forward_set.min() < value - self.bounds[self.mainSymbol]):
            if (np_forward_set.max() > value + self.bounds[self.mainSymbol]):
                # both but direction first
                if (np_forward_set.argmin() < np_forward_set.argmax()):
                    return [0,1,0]
                else:
                    return [1,0,0]
            else:
                return [0,1,0]
        return [0,0,1]

    def getStructuredData(self, dataset, orignal_set, symbol):
        x = []
        y = []
    
        # idx of new week beginnings
        week_change_idx = np.array(dataset.reset_index()['datetime'].diff() 
            > pd.Timedelta(self.maxTimeDeltaAcceptance)).nonzero()
        week_change_idx = np.append(week_change_idx, len(orignal_set))
        
        week_start_idx = 0
        for week_end_idx in np.nditer(week_change_idx):
        #    print("from: ", week_start_idx, " to: ", week_end_idx, " diff: ", week_end_idx-week_start_idx)
        #    print("next range from: ", week_start_idx+lookback_batch, " to: ", week_end_idx-forward_set_lengh)
            range_from = week_start_idx + self.lookback_batch
            range_to = week_end_idx - self.forward_set_lengh
            if range_from >= range_to:
                continue
            for i in range(range_from, range_to, self.lookback_stepsize):
                x.append(orignal_set[i-self.lookback_batch:i, 0])
                if symbol == self.mainSymbol:
                    y.append(self.getCategory(orignal_set[i], np.array(orignal_set[i+1:i+self.forward_set_lengh])))
            week_start_idx = week_end_idx
        
        return x, y

    # symbol matches directory
    def loadSymbolCSV(self, symbol):
        df = None
        
        for year in np.arange(self.beginTrain.year, self.endTest.year+1):
            for file in glob.glob("INPUT_DATA/%s/*%s*" % (symbol, year)):
                print("Load: ", file)
                next_df = pd.read_csv(file, header=None, index_col = 'datetime',
                                 parse_dates={'datetime': [0, 1]}, 
                                 date_parser=self.dateparse)
                df = pd.concat([df, next_df])
        return df

    def createScaledSet(self, trainSet, testSet):
        scaledTrainSet = []
        scaledTestSet = []
        
        sc = MinMaxScaler(feature_range = (0.05, 1))
        
        maxVal = 0
        for rangeSet in trainSet:
            newSet = rangeSet - rangeSet.min()
            scaledTrainSet.append(np.array([newSet]).reshape(-1,1))
            if newSet.max() > maxVal:
                maxVal = newSet.max()
                sc.fit_transform(newSet.reshape(-1,1))
                
        for i, rangeSet in enumerate(scaledTrainSet):
            scaledTrainSet[i] = sc.transform(rangeSet)

        for rangeSet in testSet:
            scaledTestSet.append(sc.transform(np.array([rangeSet]).reshape(-1,1)))
        
        return scaledTrainSet, scaledTestSet


    def getXYArrays(self, datasetTrain, datasetTest):
        sc = MinMaxScaler(feature_range = (0, 1))
        
        ## Main Symbol ##
        symArrTrainMain = datasetTrain[self.mainSymbol]
        training_set_main = np.array([symArrTrainMain.values]).reshape(-1,1)
        
        symArrTestMain = datasetTest[self.mainSymbol]
        test_set_main = np.array([symArrTestMain.values]).reshape(-1,1)
        
        x_arr_train_main, y_arr_train_main = self.getStructuredData(
                symArrTrainMain, training_set_main, self.mainSymbol)
        x_arr_test_main, y_arr_test_main = self.getStructuredData(
                symArrTestMain, test_set_main, self.mainSymbol)
        
        x_arr_train_main, x_arr_test_main = self.createScaledSet(x_arr_train_main, x_arr_test_main)
        
        y_train = np.array(y_arr_train_main)
        y_test = np.array(y_arr_test_main)
        
        X_train = [x_arr_train_main]
        X_test = [x_arr_test_main]
        
        # other indicator symbols
        for symbol in self.indicatorSymbols:
            symArrTrain = datasetTrain[symbol]
            training_set = np.array([symArrTrain.values]).reshape(-1,1)
            
            symArrTest = datasetTest[symbol]
            test_set = np.array([symArrTest.values]).reshape(-1,1)
        
            x_arr_train, y_arr_train = self.getStructuredData(
                    symArrTrain, training_set, symbol)
            x_arr_test, y_arr_test = self.getStructuredData(
                    symArrTest, test_set, symbol)
    
            x_arr_train, x_arr_test = self.createScaledSet(x_arr_train, x_arr_test)
    
            X_train.append(x_arr_train)
            X_test.append(x_arr_test)
    
        if self.getTrainData:
            X_train = np.array(X_train)
        else:
            X_train = []
        X_test = np.array(X_test)
        
        # Reshaping
        X_train = np.moveaxis(X_train, 0, -1)
        X_test = np.moveaxis(X_test, 0, -1)
    
        return X_train, y_train, X_test, y_test

    def getDataForSymbol(self, symbol):
        # parse 0/1 column to datetime column
        dataset_raw = None
        try:
            dataset_raw = self.loadSymbolCSV(symbol).sort_index()
        except:
            print("missing data for symbol %s for year range %s - %s, 0 rows found." % (symbol, self.beginTrain.year, self.endTest.year))
            raise
        
        dataset_inter = dataset_raw.resample('1T').asfreq().interpolate(method='quadratic', limit=self.interpolateLimit).dropna()
    
        # add BB MACD RSI here before dropna()
    
        dataset_train = dataset_inter[(dataset_inter.index > self.beginTrain) & (dataset_inter.index < self.endTrain)]
        dataset_train = dataset_train.iloc[:, 0:1]
        dataset_train = dataset_train.rename(columns = {2: symbol})
        
        dataset_test = dataset_inter[(dataset_inter.index > self.endTrain) & (dataset_inter.index < self.endTest)]
        dataset_test = dataset_test.iloc[:, 0:1]
        dataset_test = dataset_test.rename(columns = {2: symbol})
    
        return dataset_train, dataset_test