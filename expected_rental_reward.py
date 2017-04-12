from poisson import Poisson
import numpy as np


class ExpectedRentalReward(object):

    RENTAL_REWARD = 10.
    CAPACITY = 20

    @classmethod
    def get(cls, expected_request):
        return np.asarray([cls.state_reward(s, expected_request) for s in range(cls.CAPACITY + 1)])

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

