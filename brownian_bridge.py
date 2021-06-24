import numpy as np
import matplotlib.pyplot as plt 


"""
simulation of brownian bridge in order to find quantiles of test statistic
"""

n = 200
iterations = 10000
d = 2

sample = np.zeros(iterations)
steps = np.random.multivariate_normal(np.zeros(d), np.identity(d), size=(n, iterations))

for i in range(0, iterations):
	#first simulate d dimensional brownian motion
	brownian = np.zeros((d,n+1))
	for j in range(1,n+1):
		brownian[:, j] = steps[j-1, i, :] + brownian[:, j-1]
	brownian /= n**0.5

	#convert to browian bridge
	modulus = np.zeros(n+1)
	for j in range(0,n+1):
		modulus[j] = np.linalg.norm((brownian[:, j] - brownian[:, n]*(j/n)))**2;
	sample[i] = np.max(modulus)
	print(i)

np.savetxt('brownian_bridge_sample.dat', sample)
print(np.quantile(sample, 0.95))