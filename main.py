import numpy as np


"""
provides functions for calculating the proposed change point statistic, and the 
estimate of the change point, for samples from an unspecified distribution
"""

# calculates test statistic and estimate of change point for general psi
def generalised_changepoint(data, psi):
	data = np.array([psi(i) for i in data]).T
	(d, n) = np.shape(data)

	#normalises
	data -= np.ones((d, n))*np.mean(data)

	#constructs brownian motion
	brownian = np.cumsum(data, 1)
	brownian /= n**0.5

	#converts to brownian bridge
	brownian -= np.outer(brownian[:, n-1], np.arange(n)+1)/n

	#estimates cov matrix (nb np.cov gives unbiased covariance, so n-1/n needed)
	sig_inv = np.linalg.inv(np.cov(data, rowvar=True)*(n-1)/n)

	#calculates variance-normalised magnitudes of brownian bridge
	tn = np.zeros(n)
	for i in range(1, n):
		tn[i] = np.inner(brownian[:, i], np.matmul(sig_inv, brownian[:, i]))

	#t is test statistic, u is change point estimator
	t = np.max(tn)
	u = (np.argmax(tn)+1)/n
	return (t, u)

# gives first d moments of x
def moments(x, d):
	return x**(np.arange(d)+1)

# generalised test with psi as in MoM
def MoM_changepoint(data, d):
	return generalised_changepoint(data, lambda x: moments(x, d))