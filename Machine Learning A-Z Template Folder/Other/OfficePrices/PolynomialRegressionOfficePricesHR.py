# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

F,N = map(int, input().split())
train = np.array([input().split() for _ in range(N)], float)
T = int(input())
test = np.array([input().split() for _ in range(T)], float)

X = train[:, :-1]
y = train[:, -1]

regressor = LinearRegression()
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
linearPoly = regressor.fit(X_poly, y)

retVal = linearPoly.predict(poly_reg.fit_transform(test))
print(*retVal, sep='\n')
