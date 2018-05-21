# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import lag_plot

# Importing the dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values

