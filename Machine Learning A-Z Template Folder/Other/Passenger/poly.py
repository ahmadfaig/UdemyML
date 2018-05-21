import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def parse(x):
    return x[9:]
# Importing the dataset
dataset = pd.read_csv('data.csv', header = None)
X_temp = dataset.iloc[:,0].values
X = np.reshape(np.array([parse(x) for x in X_temp], int), (-1, 1))
y = dataset.iloc[:, 1].values

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
X_poly2 = X_poly[50:57,:]
y2 = y[50:57]
poly_reg.fit(X_poly, y)
regressor = LinearRegression()
regressor.fit(X_poly, y)

X_res = np.reshape(np.array([i for i in range(61,73)]), (-1, 1))
X_poly_res = poly_reg.fit_transform(X_res)

y_act = [1563178,1312558,1501793,1388316,1325942,1410769,687396,1493945,1161128,590382,1082215,1416327]

plt.plot(X, y, color = 'red')
plt.plot(X, regressor.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.plot(X_res, regressor.predict(X_poly_res), color = 'blue')
plt.plot(X_res, y_act, color = 'green')



    