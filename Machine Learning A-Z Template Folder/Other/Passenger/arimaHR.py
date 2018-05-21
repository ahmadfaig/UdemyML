import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA

N = int(input())
history = list()
for i in range(N):
    A,B = input().split()
    history.append(float(B))
    
predictions = list()

for t in range(12):
	model = ARIMA(history, order=(1,2,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	history.append(yhat)

print(*predictions, sep='\n')