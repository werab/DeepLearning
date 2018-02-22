# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from predictor.Predictor import Predictor

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