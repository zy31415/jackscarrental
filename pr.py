import numpy as np

from scipy.stats import poisson

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

n = np.asarray([ii for ii in range(11)])
pr = poisson.pmf(n, 3)

expected_re = n * 10 * pr

ax1.plot(n, pr, 'o-')

plt.xlim([0, 12])
plt.ylim([0, .25])
plt.ylabel("Probability")

ax2 = ax1.twinx()

ax2.plot(n, expected_re, 'o-r')

plt.grid('on')
plt.ylabel("Expected Reward")

plt.show()



