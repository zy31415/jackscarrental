import numpy as np
from scipy.stats import poisson


def gen_poisson_and_cutoff(max_cutoff, mu):
    p = poisson.pmf(np.arange(max_cutoff + 1), mu)
    p[]

class ExpectedRentalReward:
    mu1 = 3
    mu2 = 4

    capacity = 20

    # Poison pmf table used in computation
    poisson1 = None
    poisson2 = None

    @classmethod
    def _gen_poisson(cls):
        cls.poisson1 = poisson.pmf(np.arange(cls.capacity + 1), cls.mu1)
        cls.poisson1[-1] = 1 - poisson.cdf(cls.capacity - 1, cls.mu1)

        cls.poisson2 = poisson.pmf(np.arange(cls.capacity + 1), cls.mu2)
        cls.poisson1[-1] = 1 - poisson.cdf(cls.capacity - 1, cls.mu1)

