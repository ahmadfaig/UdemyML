import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import datetime

dataset = pd.read_csv('UsersMay2018.csv')
X = dataset.iloc[:, 0:1].values
X_temp = dataset.iloc[:, 1].values
x_date = [datetime.datetime.strptime(x, "%y-%b") for x in X_temp.tolist()]
y = dataset.iloc[:, -1].values


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

fig, ax = plt.subplots()
quarter = mdates.MonthLocator(interval = 6)   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%y-%b')
ax.xaxis.set_major_locator(quarter)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)
ax.plot(x_date, y, color = "blue")
plt.plot(x_date, regressor.predict(X), color = "red")

dataset_p = pd.read_csv('predict.csv')
P = dataset_p.iloc[:, 0:1].values
P_temp = dataset_p.iloc[:, 1].values
p_date = [datetime.datetime.strptime(x, "%y-%b") for x in P_temp.tolist()]

plt.plot(p_date, regressor.predict(P), color = "Green")
plt.show()

dataset = pd.read_csv('UsersMay2018Cum.csv')
X = dataset.iloc[:, 0:1].values
X_temp = dataset.iloc[:, 1].values
x_date = [datetime.datetime.strptime(x, "%y-%b") for x in X_temp.tolist()]
y = dataset.iloc[:, -1].values

regressor = LinearRegression()
regressor.fit(X, y)

fig, ax = plt.subplots()
quarter = mdates.MonthLocator(interval = 6)   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%y-%b')
ax.xaxis.set_major_locator(quarter)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)
ax.plot(x_date, y, color = "blue")
plt.plot(x_date, regressor.predict(X), color = "red")

dataset_p = pd.read_csv('predict.csv')
P = dataset_p.iloc[:, 0:1].values
P_temp = dataset_p.iloc[:, 1].values
p_date = [datetime.datetime.strptime(x, "%y-%b") for x in P_temp.tolist()]

plt.plot(p_date, regressor.predict(P), color = "Green")
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly[-12:],y[-12:]) 

fig, ax = plt.subplots()
quarter = mdates.MonthLocator(interval = 6)   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%y-%b')
ax.xaxis.set_major_locator(quarter)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)
ax.plot(x_date, y, color = "blue")
plt.plot(x_date[-12:], lin_reg_poly.predict(X_poly[-12:]), color = "red")

dataset_p = pd.read_csv('predict.csv')
P = dataset_p.iloc[:, 0:1].values
P_temp = dataset_p.iloc[:, 1].values
p_date = [datetime.datetime.strptime(x, "%y-%b") for x in P_temp.tolist()]

plt.plot(p_date, lin_reg_poly.predict(poly_reg.fit_transform(P)), color = "Green")
plt.show()



