from scipy.stats import poisson
import numpy as np

class ExpectedRentalReward:
    ''' I implemented two methods using different algorithms.
Tests show that loop_method is much slower than cdf_method for big s and mu,
thus cdf_method is recommended.
'''

    @classmethod
    def loop_method(cls, s, mu):
        reward = 0
        requests = 0
        
        while True:
            _p = poisson.pmf(requests, mu)
            
            _r = requests * 10 * _p if requests <= s else s * 10 * _p

            if requests > s and _r < 1e-6:
                break

            reward += _r
            requests += 1
            
        return reward

    @classmethod
    def cdf_method(cls, s, mu):
        requests = np.arange(s+1)
        rewards = 10 * np.arange(s+1)
        p = poisson.pmf(requests, mu)

        # for number of requests equal to or larger than state:
        p[-1] = 1. - poisson.cdf(s-1, mu)

        return sum(rewards * p)
