import main
import numpy as np
import matplotlib.pyplot as plt


"""
generates random samples and evaluates distribution of the p-value under the null,
to find emperical size of test 
"""

sample_size = 50

alpha = 1
theta = 1

n = 1000
test_statistic = np.zeros(n)

for i in range(0, n):
	sample = np.random.gamma(alpha,1/theta, sample_size)
	(t, u) = main.MoM_changepoint(sample, 2)
	test_statistic[i] = t

# finds p-value for each sample
data = np.loadtxt('brownian_bridge_sample.dat')
p_values = np.zeros(n)
for i in range(0, n):
	p_values[i] = (data > test_statistic[i]).sum()/len(data)

plt.hist(p_values, 5, color="darkslategrey")
plt.xlabel("p-value under null - alpha=1, gamma=1,sample size=50, iterations=10000")
plt.show()

# emperical size of 0.05 test
print((p_values <= 0.05).sum())
print((test_statistic > 2.35).sum())