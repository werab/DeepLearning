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

    def setMACD(self, df, symbol, slow, fast):
        df['slow ema'] = df[symbol].ewm(span=slow, ignore_na=False).mean()
        df['fast ema'] = df[symbol].ewm(span=fast, ignore_na=False).mean()
        df['MACD'] = (df['fast ema'] - df['slow ema'])
        
        sc = MinMaxScaler(feature_range = (-1, 1))
        df[symbol+" MACD"] = sc.fit_transform(df[['MACD']])
        
        df = df.drop(columns=['slow ema', 'fast ema', 'MACD'])
        
        return df
        
    def setSD_MA(self, df, symbol, span):
        df['ma'] = df[symbol].rolling(span).mean()
        df['sd'] = df[symbol].rolling(span).std()
        
        df = df.dropna()
        sc = MinMaxScaler(feature_range = (0, 1))
        df[symbol+" MA"] = sc.fit_transform(df[['ma']])
        df[symbol+" SD"] = sc.fit_transform(df[['sd']])
        
        df = df.drop(columns=['ma', 'sd'])
        df = df.resample('1T').asfreq()
        
        return df

    def setRSI(self, df,  symbol, period = 14):
        delta = df[symbol].diff().dropna()
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
        df[symbol+ " RSI"] = (rsi - 50) / 100

    def interp(self, df, limit):
        d = df.notna().rolling(limit + 1).agg(any).fillna(1)
        d = pd.concat({
            i: d.shift(-i).fillna(1)
            for i in range(limit + 1)
        }).prod(level=1)
    
        return df.interpolate(limit=limit, method='quadratic').where(d.astype(bool))

    def getDataForSymbol(self, symbol):
        # parse 0/1 column to datetime column
        dataset_raw = None
        try:
            dataset_raw = self.loadSymbolCSV(symbol).sort_index()
        except:
            print("missing data for symbol %s for year range %s - %s, 0 rows found." % (symbol, self.beginTrain.year, self.endTest.year))
            raise
        
        dataset_inter = dataset_raw.iloc[:, 0:1]
        dataset_inter = dataset_inter.rename(columns = {2: symbol})
        dataset_inter = dataset_inter[(dataset_inter.index > self.beginTrain) & (dataset_inter.index < self.endTest)]
        
#        with open("data.pickle", 'wb') as fp:
#            pickle.dump(dataset_raw, fp)
        
        if int(pd.__version__.split(".")[1]) < 23:
            dataset_inter = dataset_inter.resample('1T').asfreq().pipe(self.interp, 60)
        else:
            dataset_inter = dataset_inter.resample('1T').asfreq().interpolate(method='quadratic', limit=60, limit_area='inside')

        # add special data
        dataset_inter = self.setMACD(dataset_inter, symbol, 26, 12)
        dataset_inter = self.setSD_MA(dataset_inter, symbol, 20)
        self.setRSI(dataset_inter, symbol, 14)
    
        dataset_inter = dataset_inter.dropna()
    
        dataset_train = dataset_inter[(dataset_inter.index > self.beginTrain) & (dataset_inter.index < self.endTrain)]
        dataset_test = dataset_inter[(dataset_inter.index > self.endTrain) & (dataset_inter.index < self.endTest)]
    
        return dataset_train, dataset_test
    
## Test main ##
#from datetime import datetime, timedelta
#import pickle
#
#
#getTrainData = True
#endTrain = datetime(2018,1,14)
#beginTrain = endTrain - timedelta(weeks=2)
#endTest = endTrain + timedelta(weeks=2)
#
#symbol = 'EURUSD'
#
#config = {
#     'mainSymbol'             : 'EURUSD', # base lvl
#     'indicatorSymbols'       : [], # base lvl
#
#     'lookback_stepsize'      : 1, # 2nd lvl
#     'beginTrain'             : beginTrain, # 2nd lvl
#     'endTrain'               : endTrain,
#     'endTest'                : endTest, # 2nd lvl
#
#     'lookback_batch'         : 24*60, # const
#     'maxTimeDeltaAcceptance' : '1 days 1 hours', # const
#     'forward_set_lengh'      : 60, # const
#     'interpolateLimit'       : 60, # const
#     'bounds'                 : { 'EURUSD' : 0.0010 }, # const
#}
#
#
#
#dataSet = DataSet(config, True)
#trainSetRAW, testSetRAW = dataSet.getDataForSymbol(config['mainSymbol'])
#
##X_train, y_train, X_test, y_test = dataSet.getXYArrays(trainSetRAW, testSetRAW)
#
#with open('data.pickle', 'rb') as fp:
#    dataset_raw = pickle.load(fp)
#  
#def interp(df, limit):
#    d = df.notna().rolling(limit + 1).agg(any).fillna(1)
#    d = pd.concat({
#        i: d.shift(-i).fillna(1)
#        for i in range(limit + 1)
#    }).prod(level=1)
#
#    return df.interpolate(limit=limit, method='quadratic').where(d.astype(bool))
#
## todo:
## remove "interp" fix, if pandas 0.23 is live
#if int(pd.__version__.split(".")[1]) < 23:
#    dataset_inter = dataset_raw.resample('1T').asfreq().pipe(interp, 60)
#else:
#    dataset_inter = dataset_raw.resample('1T').asfreq().interpolate(method='quadratic', limit=60, limit_area='inside')
#
#
#dataset_inter = dataset_inter.iloc[:, 0:1]
#dataset_inter = dataset_inter.rename(columns = {2: symbol})

#from sklearn.preprocessing import MinMaxScaler

## MACD
##dataset_inter['26 ema'] = dataset_inter[symbol].ewm(span=26, min_periods=26, ignore_na=False).mean()
##dataset_inter['12 ema'] = dataset_inter[symbol].ewm(span=12, min_periods=12, ignore_na=False).mean()
#dataset_inter['26 ema'] = dataset_inter[symbol].ewm(span=26, ignore_na=False).mean()
#dataset_inter['12 ema'] = dataset_inter[symbol].ewm(span=12, ignore_na=False).mean()
#dataset_inter['MACD'] = (dataset_inter['12 ema'] - dataset_inter['26 ema'])
#
#sc = MinMaxScaler(feature_range = (-1, 1))
#dataset_inter["MACD scaled"] = sc.fit_transform(dataset_inter[['MACD']])
#
#
#dataset_inter['20 ma'] = dataset_inter[symbol].rolling(20).mean()
#dataset_inter['20 sd'] = dataset_inter[symbol].rolling(20).std()
#
#dataset_inter = dataset_inter.dropna()
#
#sc = MinMaxScaler(feature_range = (0, 1))
#dataset_inter["20 ma scaled"] = sc.fit_transform(dataset_inter[['20 ma']])
#dataset_inter["20 sd scaled"] = sc.fit_transform(dataset_inter[['20 sd']])
#
#period = 14
#delta = dataset_inter[symbol].diff().dropna()
#u = delta * 0
#d = u.copy()
#u[delta > 0] = delta[delta > 0]
#d[delta < 0] = -delta[delta < 0]
#u[u.index[period-1]] = np.mean( u[:period] )
#u = u.drop(u.index[:(period-1)])
#d[d.index[period-1]] = np.mean( d[:period] )
#d = d.drop(d.index[:(period-1)])
#rs = u.ewm(com=period-1, adjust=False).mean() / d.ewm(com=period-1, adjust=False).mean()
#rsi = 100 - 100 / (1 + rs)
#dataset_inter['RSI'] = rsi
#dataset_inter["RSI scaled"] = (rsi - 50) / 100
#
#
#import matplotlib.pyplot as plt
#
#plt.plot(dataset_inter["RSI scaled"], 'black')
#plt.plot(dataset_inter["MACD scaled"], 'r')
#plt.plot(dataset_inter["20 sd scaled"], 'g')
#plt.plot(dataset_inter["20 ma scaled"], 'r')
#
#plt.plot(dataset_raw.resample('1T').asfreq().iloc[:, 0:1], 'b')
#
#
#
#plt.plot(dataset_inter.iloc[:, 0:1], 'b')




















