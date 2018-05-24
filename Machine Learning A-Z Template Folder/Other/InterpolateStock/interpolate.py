import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data.csv', header=None)
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

x_known = []
prices_float = []
x_unknown = []

for i in range(len(y)):
    if not "Missing" in y[i]:
        x_known.append(i)
        prices_float.append(float(y[i]))
    else:
        x_unknown.append(i)
        
prices_known = np.array(prices_float)

from scipy import interpolate
pred_model = interpolate.UnivariateSpline(x_known, prices_known, s=100)

plt.plot(x_known,prices_known, color = "green")
plt.scatter(x_unknown, pred_model(x_unknown), color = "red")
plt.plot(x_known, pred_model(x_known), color = "blue")