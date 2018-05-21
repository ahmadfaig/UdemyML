import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA

data = pd.read_csv('data.csv', header=None)
series = pd.Series(data.iloc[:,1])

X = series.values
history = [float(x) for x in X]
history2 = [float(x) for x in X]
predictions = list()



for t in range(12):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	history.append(yhat)

plt.plot(history, color = 'green')
history2.append(1563178)
history2.append(1312558)
history2.append(1501793)
history2.append(1388316)
history2.append(1325942)
history2.append(1410769)
history2.append(687396)
history2.append(1493945)
history2.append(1161128)
history2.append(590382)
history2.append(1082215)
history2.append(1416327)
plt.plot(history2, color = 'red')
plt.plot(series.values, color = 'blue')
    