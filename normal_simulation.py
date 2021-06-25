import main
import numpy as np
import matplotlib.pyplot as plt


"""
generates sample of normal variables with changed mean and evaluates performance of the
change point estimate
"""


sample_size = 500
sig1 = 1
sig2 = 2
change_point = 0.7

n = 1000
change_point_estimates = np.zeros(n)

for i in range(0, n):
	sample = np.concatenate((np.random.normal(0, sig1, m),
							 np.random.gamma(0, sig2, sample_size-m)))
	(t, u) = main.MoM_changepoint(sample, 3)
	change_point_estimates[i] = u

plt.hist(change_point_estimates, 20, color="darkslategrey")
plt.xlabel("mu=0 normal going from sig=1 to sig=2 at 0.7, sample size=200, iterations=500")
plt.show()

