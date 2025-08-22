import numpy as np
data = np.loadtxt('../build/result/run0.csv', delimiter=',', dtype=int)
print(data.sum())