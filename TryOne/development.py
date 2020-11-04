# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
#import scipy.stats as stats
import glob
#from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sys

getTrainData = True
endTrain = datetime(2018,2,4)
beginTrain = endTrain - timedelta(weeks=2)
endTest = endTrain + timedelta(weeks=2)

symbol = 'EURUSD'

dateparse = lambda x: pd.datetime.strptime(x, '%Y.%m.%d %H:%M')

config = {
     'mainSymbol'             : 'EURUSD', # base lvl
     'indicatorSymbols'       : ['EURGBP'], # base lvl

#     'mainSymbol'             : 'EURGBP', # base lvl
#     'indicatorSymbols'       : ['EURUSD'], # base lvl

     'lookback_stepsize'      : 1, # 2nd lvl
     'beginTrain'             : beginTrain, # 2nd lvl
     'endTrain'               : endTrain,
     'endTest'                : endTest, # 2nd lvl

     'lookback_batch'         : 12*60, # const
     'maxTimeDeltaAcceptance' : '1 days 1 hours', # const
     'forward_set_lengh'      : 60, # const
     'interpolateLimit'       : 60, # const
     'bounds'                 : { 'EURUSD' : 0.0010, 'EURGBP' : 0.0010 }, # const
}

# symbol matches directory
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

    dataset_inter = dataset_inter.dropna()

    dataset_train = dataset_inter[(dataset_inter.index > beginTrain) & (dataset_inter.index < endTrain)]
    dataset_test = dataset_inter[(dataset_inter.index > endTrain) & (dataset_inter.index < endTest)]

    return dataset_train, dataset_test

def getXYArrays(datasetTrain, datasetTest):
    ## Main Symbol ##
    x_arr_train_main, y_arr_train_main, maxSet = getStructuredData(
            datasetTrain, mainSymbol, True)
    x_arr_test_main, y_arr_test_main, _ = getStructuredData(
            datasetTest, mainSymbol)
    
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

    return X_train, y_train, X_test, y_test


forward_set_lengh = config['forward_set_lengh']
bounds = config['bounds']
forward_set_lengh = config['forward_set_lengh']
lookback_stepsize = config['lookback_stepsize']

# categories
# 0: > value + bound                       --> buy 
# 1: < value - bound                       --> sell
# 2: < value + bound && > value - bound    --> nothing
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


def getStructuredData(dataset, symbol, rangeMax = False):
    x = []
    y = []

    orignal_set = np.array(dataset.loc[:,(symbol)]).reshape(-1,1)
    
    # idx of new week beginnings
    week_change_idx = np.array(dataset.reset_index()['datetime'].diff()
        > pd.Timedelta(maxTimeDeltaAcceptance)).nonzero()
    week_change_idx = np.append(week_change_idx, len(orignal_set))
    
    week_start_idx = 0
    maxSet = np.array([0])
    for week_end_idx in np.nditer(week_change_idx):
#        print("from: ", week_start_idx, " to: ", week_end_idx, " diff: ", week_end_idx-week_start_idx)
#        print("next range from: ", week_start_idx+lookback_batch, " to: ", week_end_idx-forward_set_lengh)
        range_from = week_start_idx + lookback_batch
        range_to = week_end_idx - forward_set_lengh
        if range_from >= range_to:
            continue
        
        for i in range(range_from, range_to, lookback_stepsize):
            dataRange = dataset.iloc[i-lookback_batch:i,:].to_numpy(copy=True)
            dataRange[:,0:1] = dataRange[:,0:1] - dataRange[:,0:1].min() # prepare symbol data for scaling
            # get max
            if rangeMax and dataRange[:,0:1].max() > maxSet.max():
                maxSet = dataRange[:,0:1]
            # symbol is (must be!) first column
            x.append(dataRange)
            if symbol == mainSymbol:
                y.append(getCategory(orignal_set[i], np.array(orignal_set[i+1:i+forward_set_lengh])))
        week_start_idx = week_end_idx
    
    return x, y, maxSet

def bfill_nan(arr):
    """ Backward-fill NaNs """
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]), mask.shape[0]-1)
    idx = np.minimum.accumulate(idx[::-1], axis=0)[::-1]
    out = arr[idx]
    return out

def calc_mask(arr, maxgap):
    """ Mask NaN gaps longer than `maxgap` """
    isnan = np.isnan(arr)
    cumsum = np.cumsum(isnan).astype('float')
    diff = np.zeros_like(arr)
    diff[~isnan] = np.diff(cumsum[~isnan], prepend=0)
    diff[isnan] = np.nan
    diff = bfill_nan(diff)
    return (diff <= maxgap) | ~isnan # <= instead of < compared to SO answer

mainSymbol = config['mainSymbol']
indicatorSymbols = config['indicatorSymbols']
getTrainData = True



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

#trainSetRAW, testSetRAW = getDataForSymbol(config['mainSymbol'])
#X_train, y_train, X_test, y_test = getXYArrays(trainSetRAW, testSetRAW)

# Todos:
# b() = butter
#   Df: avg(Original(30)), b(1), b(5), b(30), b(60), b(4h), b(8h)
#   getMaxTimeFrame for 8h  

#with open('data.pickle', 'rb') as fp:
#    dataset_raw = pickle.load(fp)


# def interp(df, limit):
#     d = df.notna().rolling(limit + 1).agg(any).fillna(1)
#     d = pd.concat({
#         i: d.shift(-i).fillna(1)
#         for i in range(limit + 1)
#     }).prod(level=1)

#     return df.interpolate(limit=limit, method='quadratic').where(d.astype(bool))

dataset_raw = None
try:
    dataset_raw = loadSymbolCSV(symbol).sort_index()
except:
    print("missing data for symbol %s for year range %s - %s, 0 rows found." % (symbol, beginTrain.year, endTest.year))
    raise

#plt.figure(figsize=(19,5))

import matplotlib.collections as collections
from matplotlib import dates as mdates
import matplotlib.ticker as mticker

dataset_inter = dataset_raw.iloc[:20, 0:1]
dataset_inter = dataset_inter.rename(columns = {2: symbol})

#fig, ax = plt.subplots(figsize=(19,5))

# find_gaps = dataset_inter.resample('1T').asfreq()
# dates = find_gaps.index.to_pydatetime()
# x_val = mdates.date2num(dates)

# collection = collections.BrokenBarHCollection.span_where(
#     x_val, 
#     ymin=dataset_inter[symbol].min(), 
#     ymax=dataset_inter[symbol].max(), 
#     where=find_gaps[symbol].isna(),
#     facecolor='grey',
#     edgecolor='grey', 
#     linewidth=1.0,
#     alpha=0.3,
#     label='span')
# ax.add_collection(collection)



#ax2.bar(asd.index.values, asd["NANs"],  width=0.01, alpha=0.2, color='orange')
#ax2.grid(b=False)

#asd.plot.bar(x=None, y=None, ax=ax2, alpha=0.2, color='orange')
#fig.plot(dataset_inter[symbol], 'grey', label=symbol)

#legend = plt.legend(frameon = True)
#plt.show()

#sys.exit()




#dataset_inter = dataset_inter[(dataset_inter.index > beginTrain) & (dataset_inter.index < endTest)]

# todo:
# remove "interp" fix, if pandas 0.23 is live
#if int(pd.__version__.split(".")[1]) < 23:
#dataset_inter = dataset_raw.resample('1T').asfreq().pipe(interp, 60).dropna()
#dataset_inter = dataset_inter.resample('1T').asfreq().pipe(interp, 60).dropna()
#else:
#dataset_inter = dataset_raw.resample('1T').asfreq().interpolate(method='quadratic', limit=60, limit_area='inside')

# not correct, just for development
#dataset_inter = dataset_raw.resample('1T').asfreq().interpolate(method='index', limit=60, limit_area='inside').dropna()
#dataset_inter = dataset_inter.resample('1T').asfreq().interpolate(method='index', limit=1).dropna() # <-- good enough
dataset_inter = dataset_inter.resample('1T').asfreq().interpolate(method='index', limit=60).where(calc_mask(dataset_inter[symbol],60)).dropna()
#dataset_inter = dataset_inter.resample('1T').asfreq('B').interpolate(method='index', limit=60).where(calc_mask(dataset_inter[symbol],60))

#blub = dataset_inter.resample('1T').asfreq().dropna()
#blub.plot()


#from scipy.signal import savgol_filter
from scipy.signal import filtfilt, butter

#b, a = butter(3, 0.15)
#dataset_inter["butter"] = filtfilt(b, a, dataset_inter[symbol])

#dataset_hourly = dataset_inter.resample('W').mean().dropna()

#b, a = butter(3, 0.15)
#dataset_hourly["butter"] = filtfilt(b, a, dataset_hourly[symbol])

#from scipy.signal import filtfilt, butter

#b, a = butter(3, 0.02)
#dataset_inter["butter"] = filtfilt(b, a, dataset_inter[symbol])

#dataset_inter["withNAN"] = dataset_raw.iloc[:, 0:1].fillna(10)

#plt.plot(dataset_inter[symbol], 'grey', label=symbol)
#t = ax.plot(dataset_inter[symbol], marker='.', color='grey', label=symbol)




# class Jackarow(mdates.DateFormatter):
#  	def __init__(self, fmt):
#           mdates.DateFormatter.__init__(self, fmt)
        
#  	def __call__(self, x, pos=0):
#   		# This gets called even when out of bounds, so IndexError must be prevented.
#           if dataset_inter.index.minute % 2 == 0:
#               return ''
#           else:
#               return mdates.DateFormatter.__call__(self, x, pos)

# f = ax.plot(range(len(dataset_inter)), dataset_inter[symbol], marker='.', color='grey', label=symbol) 

# ax.xaxis.set_major_locator(mticker.MaxNLocator(10))
# ax.xaxis.set_major_formatter(Jackarow('%a %H:%M'))
    

#locator = mdates.AutoDateLocator()
#formatter = mdates.AutoDateFormatter(locator)
#ax.xaxis.set_major_formatter(formatter)


#ax.set_xticklabels(dataset_inter.index.date.tolist());
#fig.autofmt_xdate()



#plt.plot(dataset_inter["RSI scaled"], 'black')
#plt.plot(dataset_inter["MACD scaled"], 'r')
#plt.plot(dataset_inter["20 sd scaled"], 'g')
#plt.plot(dataset_inter["20 ma scaled"], 'r')
#plt.plot(dataset_inter["20 sd"], 'g')
#plt.plot(dataset_inter["20 ma"], 'r')

#plt.plot(dataset_hourly[symbol], 'grey')
#plt.plot(dataset_inter[symbol], 'grey', label=symbol)

#plt.plot(dataset_inter.iloc[:, 0:1], 'y')

#plt.plot(dataset_hourly["butter"], 'red')
#plt.plot(dataset_inter["butter"], 'red', label="butter")
#b = ax.plot(dataset_inter["butter"], 'red', label="butter")
#plt.plot(dataset_inter["withNAN"], 'blue')

# plt.legend()
# plt.show()

# Reshaping
# goal shape (<events>, <timeframe 1440>, <different metrics>)
# X_train = np.moveaxis(X_train, 0, -1)
# X_test = np.moveaxis(X_test, 0, -1)

#return X_train, y_train, X_test, y_test



#a = np.array([[1.,1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09],
#              [1.1,1.11,1.12,1.13,1.14,1.15,1.16,1.17,1.18,1.19]])
#b = np.array([[2.,2.01,2.02,2.03,2.04,2.05,2.06,2.07,2.08,2.09],
#              [2.1,2.11,2.12,2.13,2.14,2.15,2.16,2.17,2.18,2.19]])
#c = np.array([[3.,3.01,3.02,3.03,3.04,3.05,3.06,3.07,3.08,3.09],
#              [3.1,3.11,3.12,3.13,3.14,3.15,3.16,3.17,3.18,3.19]])
#
#x = []
#x.append(a)
#x.append(b)
#x.append(c)
#
#d = np.array(x)
#d.shape
#np.moveaxis(d, 0, -1).shape
#
#aShape = np.moveaxis([a], 0, -1)
#bShape = np.moveaxis([b], 0, -1)
#
#np.concatenate((aShape, bShape), axis=2).shape

# shape
# (<testevents>, <timeframe 1440>, <different metrics>)


# pd.np.random.seed(1234)
# idx = pd.date_range(end=datetime.today().date(), periods=10, freq='T')
# vals = pd.Series(pd.np.random.randint(1, 10, size=idx.size), index=idx)
# vals.iloc[4:8] = pd.np.nan

# fig, ax = plt.subplots()
# ax.plot(range(vals.dropna().size), vals.dropna())
# ax.set_xticklabels(vals.dropna().index.date.tolist());
# fig.autofmt_xdate()




import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Wiggle", "Apples", "Oranges", "Bananas"],
    "Amount": [10, 1, 2, 2, 4, 10],
    "City": ["WW", "WW", "WW", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City")

app.layout = html.Div(children=[
    html.H1(children='12345 Dagreesh'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)

    print("*** End ***")





















