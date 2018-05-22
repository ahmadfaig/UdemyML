import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA

data = pd.read_csv('data.csv', header=None)
X = data.iloc[:,0].values.tolist()
history = [float(x) for x in X]
test = history[-49:]
predictions = []

for t in range(100):
	model = ARIMA(history, order=(1,1,1))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	history.append(yhat)


#data = pd.read_csv('answer.csv', header=None)
#temp = data.iloc[:,0].values.tolist()
#X2 = list(range(500,530))
#answer = [float(x) for x in temp]
plt.plot(history, color = "blue")
#plt.plot(X2,predictions, color = "green")
#plt.plot(X2,answer, color = "red")
#plt.show()
