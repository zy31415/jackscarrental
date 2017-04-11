import numpy as np
from scipy.stats import poisson


# Problem definition:
capacity = 20
discount = 0.9
rental_reward = 10.
transfer_reward = -2.

request_mean_G1 = 3
return_mean_G1 = 3
request_mean_G2 = 4
return_mean_G2 = 2


def poisson_cutoff(lam, cutoff):
    seq = np.arange(cutoff+1)
    p = poisson.pmf(seq, lam)
    # for number of requests equal to or larger than state:
    p[-1] = 1. - poisson.cdf(cutoff - 1, lam)

    return p


def expected_rental_reward(s, mu):
    global rental_reward
    p = poisson_cutoff(mu, s)
    rewards = rental_reward * np.arange(s+1)
    return sum(rewards * p)


def expected_rental_reward2D(s, mu):
    r1 = expected_rental_reward(s[0], mu[0])
    r2 = expected_rental_reward(s[1], mu[1])

    print(r1)
    print(r2)

    return r1[:, np.newaxis] + r2


def transition_probabilty(s, req, ret, action=0):
    '''
    
    :param s: Current State
    :param req: Mean value of requests
    :param ret: Mean value of returns
    :param action: Action. Positive means move in. Negativ means move out.
    :return: Transition probability.
    '''

    global capacity

    p_req = poisson_cutoff(req, s)
    p_ret = poisson_cutoff(ret, capacity - s)
    p = np.outer(p_req, p_ret)

    transp = np.zeros(capacity + 1)
    for nth, offset in enumerate(range(-s, capacity - s + 1), start=action):
        _trace = np.trace(p, offset)
        if 0 <= nth < capacity + 1:
            transp[nth] += _trace
        elif nth >= capacity + 1:
            transp[-1] += _trace
        elif nth < 0:
            transp[0] += _trace
        else:
            raise ValueError("Should not be here")

    return transp


# policy evaluation

def policy_evaluation(policy, value):

    it = np.nditer([policy, value], flags=['multi_index'])
    while not it.finished:
        p, v = it[0], it[1]

        reward = expected_rental_reward(s, mu)
        print(p, v, it.multi_index)
        it.iternext()


if __name__ == '__main__':

    # Policy to be evaluated
    policy = np.zeros([capacity + 1]*2)

    # Initial value
    value = np.zeros([capacity + 1]*2)


    policy_evaluation(policy, value)

