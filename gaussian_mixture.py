import main
import numpy as np
import multiprocessing
import time

"""
for mass testing of estimator peformance on gaussian mixtures (2 gaussians)
"""

def mixture_sample(params, weights, n):
	gaussian_choice = np.random.choice(np.arange(len(weights)), size=n, p=weights)
	return np.random.normal(params[gaussian_choice, 0], params[gaussian_choice, 1])

def estimator_performance(config):
	params = np.zeros((2, 2))
	params[0] = config[0:2]
	params[1] = config[2:4]

	weights_0 = config[4:6]
	weights_1 = config[6:8]
	change_point = config[8]
	sample_size = round(config[9])
	n = round(config[10])

	m = round(sample_size*change_point)
	change_point_estimates = np.zeros(n)

	for i in range(0, n):
		sample = np.concatenate((mixture_sample(params, weights_0, m),
							 mixture_sample(params, weights_1, sample_size-m)))
		(t, u) = main.MoM_changepoint(sample, 2)
		change_point_estimates[i] = u

	rmse = sum((change_point_estimates-change_point)**2/n)**0.5
	return (np.mean(change_point_estimates), np.var(change_point_estimates), rmse)

if __name__ == '__main__':
	configs = np.zeros((100, 11))
	for i in range(0, 100):
		configs[i] = np.array([0, 1, 1, 1, 1, 0, 0, 1, 0.5, 100, 100])

	pool = multiprocessing.Pool(16)
	performance = pool.map(estimator_performance, configs)

	name = "mixture2_data_" + str(round(time.time())) + ".txt"
	data = np.hstack((configs, performance))
	np.savetxt(name, data)