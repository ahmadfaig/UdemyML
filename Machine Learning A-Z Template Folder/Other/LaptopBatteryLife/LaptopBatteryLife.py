import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

dataset = pd.read_csv('data.csv', header=None)
dataset = dataset.sort_values(0)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

plt.scatter(X,y)
