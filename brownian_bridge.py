import numpy as np
import matplotlib.pyplot as plt 


"""
simulation of brownian bridge in order to find quantiles of test statistic
"""

n = 1000
iterations = 100000
d = 2

sample = np.zeros(iterations)
steps = np.random.multivariate_normal(np.zeros(d), np.identity(d), size=(n, iterations))

for i in range(0, iterations):
	#first simulate d dimensional brownian motion
	brownian = np.cumsum(steps[:,i,:],0).T
	brownian /= n**0.5

	#convert to browian bridge
	brownian -= np.outer(brownian[:, n-1], np.arange(n)+1)/n
	sample[i] = np.max(np.linalg.norm(brownian, axis=0))**2
	print(i)

np.savetxt('brownian_bridge_sample.dat', sample)
print(np.quantile(sample, 0.95))