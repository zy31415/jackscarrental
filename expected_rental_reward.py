from poisson import Poisson
import numpy as np


class ExpectedRentalReward(object):
    cache = None
    mu1 = None
    mu2 = None

    RENTAL_REWARD = 10.

    @classmethod
    def set(cls, mu1, mu2):
        cls.mu1 = mu1
        cls.mu2 = mu2
        cls.cache = None

    @classmethod
    def get(cls):
        if cls.cache is None:
            r1 = np.asarray([cls.state_reward(s, cls.mu1) for s in range(21)])
            r2 = np.asarray([cls.state_reward(s, cls.mu2) for s in range(21)])
            cls.cache = r1[:, np.newaxis] + r2

        return cls.cache

    @classmethod
    def state_reward(cls, s, mu):
        rewards = cls.RENTAL_REWARD * np.arange(s + 1)
        p = Poisson.pmf_series(mu, cutoff=s)
        return sum(rewards * p)


if __name__ == '__main__':

    ExpectedRentalReward.set(3, 4)

    r = ExpectedRentalReward.get()

    print(r)

    import matplotlib.pylab as plt

    plt.pcolor(r)
    plt.colorbar()
    plt.show()

