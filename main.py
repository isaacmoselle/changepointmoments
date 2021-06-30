import numpy as np
import matplotlib.pyplot as plt
import time


"""
provides functions for calculating the proposed change point statistic, and the 
estimate of the change point, for samples from an unspecified distribution
"""

# calculates test statistic and estimate of change point for general data, post psi
def generalised_changepoint(data):
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
	tn = np.einsum('ij,ij -> j', brownian, np.matmul(sig_inv,brownian))

	#t is test statistic, u is change point estimator
	t = np.max(tn)
	u = (np.argmax(tn)+1)/n
	return (t, u)

# generalised test with psi as in MoM
def MoM_changepoint(data, d):
	#computes first d moments of data
	data = np.vstack([data**i for i in np.arange(d) + 1 ])
	return generalised_changepoint(data)