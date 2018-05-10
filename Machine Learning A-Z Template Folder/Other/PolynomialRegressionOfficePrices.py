# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model as lm
from sklearn import preprocessing as pp

dataset = pd.read_csv('data.csv')

X_train = dataset.iloc[:,0]
y_train = dataset.iloc[:,-1]

plt.scatter(X_train, y_train)
plt.show()