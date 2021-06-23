import main
import numpy as np
import matplotlib.pyplot as plt


"""
generates sample of normal variables with changed mean and evaluates performance of the
change point estimate
"""


sample_size = 100
m1 = 0
m2 = 1
change_point = 0.2

n = 500
change_point_estimates = np.zeros(n)

for i in range(0, n):
	sample = np.zeros(sample_size)
	for j in range(0, round(sample_size*change_point)):
		sample[j] = np.random.normal(m1, 1)
	for j in range(round(sample_size*change_point)+1, sample_size):
		sample[j] = np.random.normal(m2, 1)

	(p, u) = main.MoM_changepoint(sample, 2)
	change_point_estimates[i] = u

plt.hist(change_point_estimates, 20)
plt.show()