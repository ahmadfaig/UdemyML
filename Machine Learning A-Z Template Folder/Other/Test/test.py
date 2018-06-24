import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data.csv',header=None)
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 1].values.tolist()
#
#dataset = pd.read_csv('result.csv',header=None)
#X_2 = dataset.iloc[:, 0:1].values
#y_2 = dataset.iloc[:, 1].values

count = 120
from statsmodels.tsa.arima_model import ARIMA
predictions = []
history = [float(x) for x in X]

#if len(history) < 240:
#    temp = history[:]
#    for i in range(10):
#        history = history + temp

for t in range(count):
	model = ARIMA(history, order=(5,0,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0][0]
	predictions.append(yhat)
	history.append(yhat)

plt.plot(history)
