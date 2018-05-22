import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA

N = int(input())
history = list()
for i in range(N):
    A = input()
    history.append(float(A))

test = history[70:101]
predictions = []

for t in range(30):
	model = ARIMA(test, order=(1,0,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	test.append(yhat)
    
print(*predictions, sep='\n')

