# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import glob
import scipy.stats as stats
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

    def getStructuredData(self, dataset, symbol, rangeMax = False):
        x = []
        y = []
    
        orignal_set = np.array(dataset.loc[:,(symbol)]).reshape(-1,1)
        
        # idx of new week beginnings
        week_change_idx = np.array(dataset.reset_index()['datetime'].diff() 
            > pd.Timedelta(self.maxTimeDeltaAcceptance)).nonzero()
        week_change_idx = np.append(week_change_idx, len(orignal_set))
        
        week_start_idx = 0
        maxSet = np.array([0])
        for week_end_idx in np.nditer(week_change_idx):
    #        print("from: ", week_start_idx, " to: ", week_end_idx, " diff: ", week_end_idx-week_start_idx)
    #        print("next range from: ", week_start_idx+lookback_batch, " to: ", week_end_idx-forward_set_lengh)
            range_from = week_start_idx + self.lookback_batch
            range_to = week_end_idx - self.forward_set_lengh
            if range_from >= range_to:
                continue
            
            for i in range(range_from, range_to, self.lookback_stepsize):
                dataRange = dataset.iloc[i-self.lookback_batch:i,:].as_matrix().copy()
                dataRange[:,0:1] = dataRange[:,0:1] - dataRange[:,0:1].min() # prepare symbol data for scaling
                # get max
                if rangeMax and dataRange[:,0:1].max() > maxSet.max():
                    maxSet = dataRange[:,0:1]
                # symbol is (must be!) first column
                x.append(dataRange)
                if symbol == self.mainSymbol:
                    y.append(self.getCategory(orignal_set[i], np.array(orignal_set[i+1:i+self.forward_set_lengh])))
            week_start_idx = week_end_idx
        
        return x, y, maxSet

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

    def getXYArrays(self, datasetTrain, datasetTest):
        ## Main Symbol ##
        x_arr_train_main, y_arr_train_main, maxSet = self.getStructuredData(
                datasetTrain, self.mainSymbol, True)
        x_arr_test_main, y_arr_test_main, _ = self.getStructuredData(
                datasetTest, self.mainSymbol)
        
        # scaling main symbol
        sc = MinMaxScaler(feature_range = (0.05, 1))
        sc.fit_transform(maxSet)
        
        for i, rangeSet in enumerate(x_arr_train_main):
            x_arr_train_main[i][:,0:1] = sc.transform(rangeSet[:,0:1])
            
        for i, rangeSet in enumerate(x_arr_test_main):
            x_arr_test_main[i][:,0:1] = sc.transform(rangeSet[:,0:1])
        
        
        X_train = np.array(x_arr_train_main)
        X_test = np.array(x_arr_test_main)
        
        y_train = np.array(y_arr_train_main)
        y_test = np.array(y_arr_test_main)
        
        
        # other indicator symbols
        for symbol in self.indicatorSymbols:
            train, test = self.getDataForSymbol(symbol)
            
            x_arr_train, y_arr_train, maxSet = self.getStructuredData(
                train, symbol, True)
            x_arr_test, y_arr_test, _ = self.getStructuredData(
                test, symbol)
            
            # scaling symbol
            sc.fit_transform(maxSet)
        
            for i, rangeSet in enumerate(x_arr_train):
                x_arr_train[i][:,0:1] = sc.transform(rangeSet[:,0:1])
                
            for i, rangeSet in enumerate(x_arr_test):
                x_arr_test[i][:,0:1] = sc.transform(rangeSet[:,0:1])
            
            X_train = np.concatenate((X_train, np.array(x_arr_train)), axis=2)
            X_test = np.concatenate((X_test, np.array(x_arr_test)), axis=2)
    
        return X_train, y_train, X_test, y_test

    def setMACD(self, df, symbol, slow, fast):
        df['slow ema'] = df[symbol].ewm(span=slow, ignore_na=False).mean()
        df['fast ema'] = df[symbol].ewm(span=fast, ignore_na=False).mean()
        df['MACD'] = (df['fast ema'] - df['slow ema'])
        
        sc = MinMaxScaler(feature_range = (0, 1))
        
        # save indizes with negatives
        neg_idx = (df['MACD'] < 0)
        # multiply indexed values with -1
        df['MACD'][neg_idx] = df['MACD']*-1
        # scale
        df['MACD'] = stats.mstats.winsorize(df['MACD'].values, limits=(None, 0.001))
        df[symbol+" MACD"] = sc.fit_transform(df[['MACD']])
        # multiply indexed values with -1
        df[symbol+" MACD"][neg_idx] = df[symbol+" MACD"]*-1
        
        df = df.drop(columns=['slow ema', 'fast ema', 'MACD'])
        
        return df

    def setSD_MA(self, df, symbol, span):
        df['ma'] = df[symbol].rolling(span).mean()
        df['sd'] = df[symbol].rolling(span).std()
        
        df = df.dropna()
        df.is_copy = False
        
        df['sd'] = stats.mstats.winsorize(df['sd'].values, limits=(None, 0.001))
        
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
        df[symbol+ " RSI"] = (rsi - 50)*2 / 100

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
#     'indicatorSymbols'       : ['EURGBP'], # base lvl
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
#for sym in config['indicatorSymbols']:
#    _train, _test = dataSet.getDataForSymbol(sym)
#
#    trainSetRAW = pd.concat([trainSetRAW, _train], axis=1, join_axes=[trainSetRAW.index])
#    testSetRAW = pd.concat([testSetRAW, _test], axis=1, join_axes=[testSetRAW.index])
#
#X_train, y_train, X_test, y_test = dataSet.getXYArrays(trainSetRAW, testSetRAW)
#
#trainSetRAW, testSetRAW = dataSet.getDataForSymbol(config['mainSymbol'])



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




















