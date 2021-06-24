import main
import numpy as np
import matplotlib.pyplot as plt


"""
generates sample of gamma variables evaluates performance of the
change point estimate
"""


sample_size = 500
change_point = 0.75

alpha1 = 1
alpha2 = 1
theta1 = 0.01
theta2 = 0.05

n = 1000
change_point_estimates = np.zeros(n)

for i in range(0, n):
	sample = np.zeros(sample_size)
	for j in range(0, round(sample_size*change_point)):
		sample[j] = np.random.gamma(alpha1,1/theta1)
	for j in range(round(sample_size*change_point)+1, sample_size):
		sample[j] = np.random.gamma(alpha2,1/theta2)
	(p, u) = main.MoM_changepoint(sample, 2)
	change_point_estimates[i] = u
	print(i)

plt.hist(change_point_estimates, 30, color="darkslategrey")
plt.xlabel("alpha=1 gamma going from theta=0.01 to theta=0.05 at 0.75, sample size=500, iterations=1000")
plt.show()

print(np.average(change_point_estimates))
print(np.std(change_point_estimates))
print(sum((change_point_estimates-0.75)**2/n)**0.5)