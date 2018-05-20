# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

#F,N = map(int, input().split())
#train = np.array([input().split() for _ in range(N)], float)
#T = int(input())
#test = np.array([input().split() for _ in range(T)], float)

dataset = pd.read_csv('data.csv', header=None)
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:,-1].values

regressor = LinearRegression()
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
linearPoly = regressor.fit(X_poly, y)

test = [[0.05, 0.54],[0.91,0.91],[0.31,0.76],[0.51,0.31]]

retVal = linearPoly.predict(poly_reg.fit_transform(test))
print(*retVal, sep='\n')
