import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.csv',header=None)
dataset_test = pd.read_csv('test.csv',header=None)
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2].values
X_test = dataset_test.iloc[:, 0:2].values

#F,N = map(int, input().split())
#train = np.array([input().split() for _ in range(N)], float)
#T = int(input())
#test = np.array([input().split() for _ in range(T)], float)
#
#X = train[:, :-1]
#y = train[:, -1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

y_pred = regressor.predict(X_test)

