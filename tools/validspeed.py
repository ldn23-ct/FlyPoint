import numpy as np
import matplotlib.pyplot as plt
dir = "./TrueData/test/2025-11-17_15-33_50eda15f"
file = dir + "/0_events_with_angle.npy"
data = np.load(file)[20000:, [0, 3]]

coef = np.polyfit(data[:, 0], data[:, 1], 1)
a, b = coef
x_fit = a * data[:, 0] + b
residual = data[:, 1] - x_fit

factor = 100
t_plot = data[:, 0][::factor]
x_plot = data[:, 1][::factor]
x_fit_plot = (a * t_plot + b)
plt.figure()
plt.plot(t_plot, x_plot, '.', markersize=1, label='raw (downsampled)')
plt.plot(t_plot, x_fit_plot, '-', label='linear fit')
plt.legend()
plt.title('x-t trend')

r_plot = residual[::factor]
plt.figure()
plt.plot(t_plot, r_plot, '.', markersize=1)
plt.title('residual = x - (a t + b)')
plt.show()
