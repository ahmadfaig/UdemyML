import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data.csv')
x = dataset.iloc[:, 0].values
tmax = dataset.iloc[:, 2].values
tmin = dataset.iloc[:, 3].values

#N = int(input())
#temp = input()
#
#tmax = []
#tmin = []
#for i in range(N):
#    a,b,c,d = input().split("\t")
#    tmax.append(c)
#    tmin.append(d)

x_tmax_known=[]
y_tmax_known=[]
x_tmax_unknown=[]
tmax_result=[]

x_tmin_known=[]
y_tmin_known=[]
x_tmin_unknown=[]
tmin_result=[]

for i in range(len(tmax)):
    if not "Missing" in tmax[i]:
        x_tmax_known.append(i)
        y_tmax_known.append(float(tmax[i]))
    else:
        x_tmax_unknown.append(i)
    if not "Missing" in tmin[i]:
        x_tmin_known.append(i)
        y_tmin_known.append(float(tmin[i]))
    else:
        x_tmin_unknown.append(i)
        

from scipy import interpolate
pred_model_tmax = interpolate.UnivariateSpline(x_tmax_known, y_tmax_known, s=0)
pred_model_tmin = interpolate.UnivariateSpline(x_tmin_known, y_tmin_known, s=10)

tmax_result = np.column_stack((x_tmax_unknown,pred_model_tmax(x_tmax_unknown)))
tmin_result = np.column_stack((x_tmin_unknown,pred_model_tmin(x_tmin_unknown)))

#plt.plot(x_tmax_known[0:110],y_tmax_known[0:110],color = "blue")
#plt.plot(x_tmax_known[0:100],pred_model_tmax(x_tmax_known[0:100]),color = "green")
#plt.scatter(x_tmax_unknown[0:20],pred_model_tmax(x_tmax_unknown[0:20]),color = "red")
#plt.show()
#plt.plot(x_tmin_known,y_tmin_known,color = "blue")
#plt.scatter(x_tmin_unknown,pred_model_tmin(x_tmin_unknown),color = "red")
#plt.show()

retVal = np.concatenate([tmax_result,tmin_result])
retVal = retVal[retVal[:,0].argsort()]

for i in range(len(retVal)):
    print(retVal[i,1])

