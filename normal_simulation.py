import main
import numpy as np
import matplotlib.pyplot as plt


"""
generates sample of normal variables with changed mean and evaluates performance of the
change point estimate
"""


sample_size = 200
sig1 = 1
sig2 = 2
change_point = 0.7

n = 500
change_point_estimates = np.zeros(n)

for i in range(0, n):
	sample = np.zeros(sample_size)
	for j in range(0, round(sample_size*change_point)):
		sample[j] = np.random.normal(0,sig1**0.5)
	for j in range(round(sample_size*change_point)+1, sample_size):
		sample[j] = np.random.normal(0,sig2**0.5)

	(p, u) = main.MoM_changepoint(sample, 2)
	change_point_estimates[i] = u

plt.hist(change_point_estimates, 20, color="darkslategrey")
plt.xlabel("mu=0 normal going from sig=1 to sig=2 at 0.7, sample size=200, iterations=500")
plt.show()