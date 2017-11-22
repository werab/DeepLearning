# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# params
lookback_batch = 24*60
lookback_stepsize = 2

# Importing the training set
dataset_train = pd.read_csv('DAT_MT_EURUSD_M1_201710.csv', header=None)
training_set = dataset_train.iloc[:, 2:3].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []

setSize = len(training_set)

for i in range(lookback_batch, setSize, lookback_stepsize):
    X_train.append(training_set_scaled[i-lookback_batch:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

myX_train, myY_train = np.array(X_train), np.array(y_train)
myX_train.shape[1]

# Reshaping
myX_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))