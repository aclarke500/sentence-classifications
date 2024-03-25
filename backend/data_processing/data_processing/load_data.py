import pandas as pd
import numpy as np

x_test = pd.read_csv('../data/X_test.csv')
# y_test = pd.read_csv('../data/y_test.csv')[0].values
y_test = np.genfromtxt('../data/y_test.csv', delimiter=',', skip_header=True)
y_train = np.genfromtxt('../data/y_train.csv', delimiter=',', skip_header=True)
print(y_test)