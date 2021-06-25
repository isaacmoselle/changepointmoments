import main
import numpy as np
import matplotlib.pyplot as plt


"""
generates sample of gamma variables evaluates performance of the change point estimate
"""


sample_size = 500
change_point = 0.75

alpha1 = 1
alpha2 = 1
lamb1 = 0.01
lamb2 = 0.05

n = 1000
change_point_estimates = np.zeros(n)
m = round(sample_size*change_point)

for i in range(0, n):
	sample = np.concatenate((np.random.gamma(alpha1, 1/lamb1, m),
							 np.random.gamma(alpha2, 1/lamb2, sample_size-m)))
	(t, u) = main.MoM_changepoint(sample, 2)
	change_point_estimates[i] = u

plt.hist(change_point_estimates, 30, color="darkslategrey")
plt.xlabel("alpha=1 gamma going from theta=0.01 to theta=0.05 at 0.75, sample size=500, iterations=1000")
plt.show()

print(np.average(change_point_estimates))
print(np.std(change_point_estimates))
print(sum((change_point_estimates-0.75)**2/n)**0.5)