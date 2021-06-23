import numpy as np

"""
provides functions for calculating the proposed change point statistic, and the 
estimate of the change point, for samples from an unspecified distribution
"""


# calculates first d sample moments
def moments(data, d):
	moments = np.zeros(d)
	for i in range(0, d):
		moments[i] = np.average(data**(i+1))
	return moments


# calculates Zn(u, theta^) as in the paper, where theta^ is the MoM estimator
# (does not need to be found), for 0<=u<=1 in intervals of 1/n
def Zn_estimator(data, d):
	n = len(data)
	sample_moments = moments(data, d)

	#holds difference between the moments for each element and the average moments
	delta = np.zeros((d, n))
	for k in range(0, n):
		# gives vector with first d powers of kth member of sample
		delta[:, k] = data[k]**(np.arange(d) + 1) -  sample_moments

	Zn = np.zeros((d, n+1))
	for k in range(1, n+1):
		#computes partial sums
		Zn[:,k] = np.sum(delta[:,:k], axis=1)/n
	return Zn

	

# calculates estimate of moment covariance matrix
def cov_estimate(data, d):
	n = len(data)
	sig = np.zeros((d, d))
	sample_moments = moments(data, d)

	for k in range(0, n):
		# gives vector with first d powers of kth member of sample
		v = data[k]**(np.arange(d) + 1) -  sample_moments
		sig += np.outer(v, v)
	return sig/n


# calculates Tn(u) as in the paper for 0<=u<=1 in intervals of 1/n
def Tn(data, d):
	n = len(data)
	sample_moments = moments(data, d)


	#first Zn(u,theta^) calculated, where theta^ is MoM estimator (need not be found)
	Zn = np.zeros((d, n+1))

	#holds difference between the moments for each element and the average moments
	delta = np.zeros((d, n))
	for k in range(0, n):
		delta[:, k] = data[k]**(np.arange(d) + 1) -  sample_moments

	for k in range(1, n+1):
		#computes partial sums
		Zn[:,k] = np.sum(delta[:,:k], axis=1)/n


	sig_inv = np.linalg.inv(cov_estimate(data, d))


	Tn = np.zeros(n+1)
	for i in range(0,n+1):
		Tn[i] = n*np.matmul(Zn[:,i].T,np.matmul(sig_inv,Zn[:,i]))
	return Tn;


# calculates test statistic and estimate of change point
def MoM_changepoint(data, d):
	tn = Tn(data, d)
	t = np.max(tn)
	u = np.argmax(tn)/len(data)
	return (t, u)