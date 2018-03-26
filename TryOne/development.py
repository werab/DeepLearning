# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import glob
#from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

getTrainData = True
endTrain = datetime(2018,2,4)
beginTrain = endTrain - timedelta(weeks=2)
endTest = endTrain + timedelta(weeks=2)

symbol = 'EURUSD'

dateparse = lambda x: pd.datetime.strptime(x, '%Y.%m.%d %H:%M')

config = {
     'mainSymbol'             : 'EURUSD', # base lvl
     'indicatorSymbols'       : ['EURGBP'], # base lvl

     'lookback_stepsize'      : 1, # 2nd lvl
     'beginTrain'             : beginTrain, # 2nd lvl
     'endTrain'               : endTrain,
     'endTest'                : endTest, # 2nd lvl

     'lookback_batch'         : 24*60, # const
     'maxTimeDeltaAcceptance' : '1 days 1 hours', # const
     'forward_set_lengh'      : 60, # const
     'interpolateLimit'       : 60, # const
     'bounds'                 : { 'EURUSD' : 0.0010 }, # const
}

def loadSymbolCSV(symbol):
    df = None
    
    for year in np.arange(beginTrain.year, endTest.year+1):
        for file in glob.glob("INPUT_DATA/%s/*%s*" % (symbol, year)):
            print("Load: ", file)
            next_df = pd.read_csv(file, header=None, index_col = 'datetime',
                             parse_dates={'datetime': [0, 1]}, 
                             date_parser=dateparse)
            df = pd.concat([df, next_df])
    return df

def interp(df, limit):
    d = df.notna().rolling(limit + 1).agg(any).fillna(1)
    d = pd.concat({
        i: d.shift(-i).fillna(1)
        for i in range(limit + 1)
    }).prod(level=1)

    return df.interpolate(limit=limit, method='quadratic').where(d.astype(bool))

def setMACD(df, symbol, slow, fast):
    df['slow ema'] = df[symbol].ewm(span=slow, ignore_na=False).mean()
    df['fast ema'] = df[symbol].ewm(span=fast, ignore_na=False).mean()
    df['MACD'] = (df['fast ema'] - df['slow ema'])
    
    sc = MinMaxScaler(feature_range = (-1, 1))
    df[symbol+" MACD"] = sc.fit_transform(df[['MACD']])
    
    df = df.drop(columns=['slow ema', 'fast ema', 'MACD'])
    
    return df
    
def setSD_MA(df, symbol, span):
    df['ma'] = df[symbol].rolling(span).mean()
    df['sd'] = df[symbol].rolling(span).std()
    
    df = df.dropna()
    sc = MinMaxScaler(feature_range = (0, 1))
    df[symbol+" MA"] = sc.fit_transform(df[['ma']])
    df[symbol+" SD"] = sc.fit_transform(df[['sd']])
    
    df = df.drop(columns=['ma', 'sd'])
    df = df.resample('1T').asfreq()
    
    return df

def setRSI(df,  symbol, period = 14):
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

def getDataForSymbol(symbol):
    # parse 0/1 column to datetime column
    dataset_raw = None
    try:
        dataset_raw = loadSymbolCSV(symbol).sort_index()
    except:
        print("missing data for symbol %s for year range %s - %s, 0 rows found." % (symbol, beginTrain.year, endTest.year))
        raise
    
    dataset_inter = dataset_raw.iloc[:, 0:1]
    dataset_inter = dataset_inter.rename(columns = {2: symbol})
    dataset_inter = dataset_inter[(dataset_inter.index > beginTrain) & (dataset_inter.index < endTest)]
    
    if int(pd.__version__.split(".")[1]) < 23:
        dataset_inter = dataset_inter.resample('1T').asfreq().pipe(interp, 60)
    else:
        dataset_inter = dataset_inter.resample('1T').asfreq().interpolate(method='quadratic', limit=60, limit_area='inside')

    # add special data
    dataset_inter = setMACD(dataset_inter, symbol, 26, 12)
    dataset_inter = setSD_MA(dataset_inter, symbol, 20)
    setRSI(dataset_inter, symbol, 14)

    dataset_inter = dataset_inter.dropna()

    dataset_train = dataset_inter[(dataset_inter.index > beginTrain) & (dataset_inter.index < endTrain)]
    dataset_test = dataset_inter[(dataset_inter.index > endTrain) & (dataset_inter.index < endTest)]

    return dataset_train, dataset_test

forward_set_lengh = config['forward_set_lengh']
bounds = config['bounds']
forward_set_lengh = config['forward_set_lengh']
lookback_stepsize = config['lookback_stepsize']

def getCategory(value, np_forward_set):
    if (np_forward_set.max() > value + bounds[mainSymbol]):
        if (np_forward_set.min() < value - bounds[mainSymbol]):
            # both but direction first
            if (np_forward_set.argmin() < np_forward_set.argmax()):
                return [0,1,0]
            else:
                return [1,0,0]
        else:
            return [1,0,0]
    elif (np_forward_set.min() < value - bounds[mainSymbol]):
        if (np_forward_set.max() > value + bounds[mainSymbol]):
            # both but direction first
            if (np_forward_set.argmin() < np_forward_set.argmax()):
                return [0,1,0]
            else:
                return [1,0,0]
        else:
            return [0,1,0]
    return [0,0,1]

maxTimeDeltaAcceptance = config['maxTimeDeltaAcceptance']
lookback_batch = config['lookback_batch']
forward_set_lengh = config['forward_set_lengh']
lookback_stepsize = config['lookback_stepsize']

def getStructuredData(dataset, symbol):
    x = []
    y = []

    orignal_set = np.array(dataset[[symbol]]).reshape(-1,1)
    
    # idx of new week beginnings
    week_change_idx = np.array(dataset.reset_index()['datetime'].diff() 
        > pd.Timedelta(maxTimeDeltaAcceptance)).nonzero()
    week_change_idx = np.append(week_change_idx, len(orignal_set))
    
    week_start_idx = 0
    for week_end_idx in np.nditer(week_change_idx):
    #    print("from: ", week_start_idx, " to: ", week_end_idx, " diff: ", week_end_idx-week_start_idx)
    #    print("next range from: ", week_start_idx+lookback_batch, " to: ", week_end_idx-forward_set_lengh)
        range_from = week_start_idx + lookback_batch
        range_to = week_end_idx - forward_set_lengh
        if range_from >= range_to:
            continue
        for i in range(range_from, range_to, lookback_stepsize):
            # symbol must be first column
            x.append(dataset.iloc[i-lookback_batch:i,:].as_matrix())
            if symbol == mainSymbol:
                y.append(getCategory(orignal_set[i], np.array(orignal_set[i+1:i+forward_set_lengh])))
        week_start_idx = week_end_idx
    
    return x, y

def createScaledSet(trainSet, testSet):
    scaledTrainSet = []
    scaledTestSet = []
    
    sc = MinMaxScaler(feature_range = (0.05, 1))
    
    maxSet = None    
    for rangeSet in trainSet:
        newSet = rangeSet - rangeSet.min()
        scaledTrainSet.append(np.array([newSet]).reshape(-1,1))
        if newSet.max() > maxSet.max():
            maxSet = newSet
            
    sc.fit_transform(maxSet.reshape(-1,1))
            
    for i, rangeSet in enumerate(scaledTrainSet):
        scaledTrainSet[i] = sc.transform(rangeSet)

    for rangeSet in testSet:
        scaledTestSet.append(sc.transform(np.array([rangeSet]).reshape(-1,1)))
    
    return scaledTrainSet, scaledTestSet

mainSymbol = config['mainSymbol']
indicatorSymbols = config['indicatorSymbols']
getTrainData = True



trainSetRAW, testSetRAW = getDataForSymbol(config['mainSymbol'])

#for sym in config['indicatorSymbols']:
#    _train, _test = getDataForSymbol(sym)
#
#    trainSetRAW = pd.concat([trainSetRAW, _train], axis=1, join_axes=[trainSetRAW.index])
#    testSetRAW = pd.concat([testSetRAW, _test], axis=1, join_axes=[testSetRAW.index])

#X_train, y_train, X_test, y_test = getXYArrays(trainSetRAW, testSetRAW)

datasetTrain = trainSetRAW.copy()
datasetTest = testSetRAW.copy()

orignal_set = np.array(datasetTrain[[symbol]]).reshape(-1,1)

# delete symbol column
datasetTrain.drop(columns=[symbol], inplace=True)

x = []
for i in range(3):
    symbolSet = orignal_set[0+i:10+i, :]
    calcSet = datasetTrain.iloc[0+i:10+i,:].as_matrix()
    x.append(np.concatenate((symbolSet, calcSet), axis=1))
    
    
bla = np.array(x).shape

#def getXYArrays(datasetTrain, datasetTest):
    ## Main Symbol ##
#symArrTrainMain = datasetTrain[mainSymbol] # change
#training_set_main = np.array(datasetTrain[[mainSymbol]]).reshape(-1,1)

#symArrTestMain = datasetTest[mainSymbol] # change
#test_set_main = np.array(datasetTest[[mainSymbol]]).reshape(-1,1)

# todo sturctured data pandas df must return np.shape (<events>, <timeframe 1440>, <different metrics>)
# test scaling with one (main) metric
x_arr_train_main, y_arr_train_main = getStructuredData(
        datasetTrain, mainSymbol)
x_arr_test_main, y_arr_test_main = getStructuredData(
        datasetTest, mainSymbol)

x_arr_train_main, x_arr_test_main = createScaledSet(x_arr_train_main, x_arr_test_main)

np.array(x_arr_train_main).shape

b = np.squeeze(np.array(x_arr_train_main))

y_train = np.array(y_arr_train_main)
y_test = np.array(y_arr_test_main)

X_train = [x_arr_train_main]
X_test = [x_arr_test_main]

# other indicator symbols
for symbol in indicatorSymbols:
    symArrTrain = datasetTrain[symbol]
    training_set = np.array([symArrTrain.values]).reshape(-1,1)
    
    symArrTest = datasetTest[symbol]
    test_set = np.array([symArrTest.values]).reshape(-1,1)

    x_arr_train, y_arr_train = getStructuredData(
            symArrTrain, training_set, symbol)
    x_arr_test, y_arr_test = getStructuredData(
            symArrTest, test_set, symbol)

    x_arr_train, x_arr_test = createScaledSet(x_arr_train, x_arr_test)

    X_train.append(x_arr_train)
    X_test.append(x_arr_test)

if getTrainData:
    X_train = np.array(X_train)
else:
    X_train = []
X_test = np.array(X_test)

# Reshaping
# goal shape (<events>, <timeframe 1440>, <different metrics>)
X_train = np.moveaxis(X_train, 0, -1)
X_test = np.moveaxis(X_test, 0, -1)

#return X_train, y_train, X_test, y_test



a = np.array([[1.,1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09],
              [1.1,1.11,1.12,1.13,1.14,1.15,1.16,1.17,1.18,1.19]])
b = np.array([[2.,2.01,2.02,2.03,2.04,2.05,2.06,2.07,2.08,2.09],
              [2.1,2.11,2.12,2.13,2.14,2.15,2.16,2.17,2.18,2.19]])
c = np.array([[3.,3.01,3.02,3.03,3.04,3.05,3.06,3.07,3.08,3.09],
              [3.1,3.11,3.12,3.13,3.14,3.15,3.16,3.17,3.18,3.19]])

x = []
x.append(a)
x.append(b)
x.append(c)

d = np.array(x)
d.shape
np.moveaxis(d, 0, -1).shape

aShape = np.moveaxis([a], 0, -1)
bShape = np.moveaxis([b], 0, -1)

np.concatenate((aShape, bShape), axis=2).shape

# shape
# (<testevents>, <timeframe 1440>, <different metrics>)




