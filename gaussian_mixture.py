import main
import numpy as np
import matplotlib.pyplot as plt

def mixture_sample(params, weights, n):
	gaussian_choice = np.random.choice(np.arange(len(weights)), size=n, p=weights)
	return np.random.normal(params[gaussian_choice, 0], params[gaussian_choice, 1])

params = np.array([[0, 1], [1, 1]])

weights_0 = np.array([0.2, 0.8])
weights_1 = np.array([0.8, 0.2])

sample_size = 500
change_point = 0.5

n = 1000

m = round(sample_size*change_point)
change_point_estimates = np.zeros(n)

for i in range(0, n):
	sample = np.concatenate((mixture_sample(params, weights_0, m),
							 mixture_sample(params, weights_1, sample_size-m)))
	(t, u) = main.MoM_changepoint(sample, 2)
	change_point_estimates[i] = u

rmse = sum((change_point_estimates-change_point)**2/n)**0.5
print(rmse)