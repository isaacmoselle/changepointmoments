import numpy as np
import matplotlib.pyplot as plt 
import time

"""
simulation of brownian bridge in order to find quantiles of test statistic
"""

n = 10
iterations = 50000
d = 2

sample = np.zeros(iterations)
steps = np.random.multivariate_normal(np.zeros(d), np.identity(d), size=(n, iterations))

for i in range(0, iterations):
	#first simulate d dimensional brownian motion
	brownian = np.zeros((d,n+1))
	for j in range(1,n+1):
		brownian[:, j] = np.sum(steps[:j, i, :])/n**0.5
	#convert to browian bridge
	for j in range(1,n+1):
		brownian[:, j] = np.linalg.norm((brownian[:, j] - brownian[:, n]*(j/n)));
	sample[i] = np.max(brownian)

plt.hist(sample, 50)
plt.show()