import numpy
import scipy.interpolate
import random
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('data.csv', header=None)
num = dataset.iloc[:, 1].values.tolist()

a = numpy.array(num)
counts, bins = numpy.histogram(a, bins=10, density=True)
cum_counts = numpy.cumsum(counts)
bin_widths = (bins[1:] - bins[:-1])

x = cum_counts*bin_widths
y = bins[1:]
inverse_density_function = scipy.interpolate.interp1d(x, y)
b = numpy.zeros(60+12)

for i in range(len( b )):
    u = random.uniform( x[0], x[-1] )
    b[i] = inverse_density_function( u )
    
solution=[]
for i in range(12):
    u = random.uniform( x[0], x[-1] )
    solution.append(inverse_density_function(u))


plt.plot(list(range(0,60)),num, color = "red")
plt.plot(list(range(0,72)),b)