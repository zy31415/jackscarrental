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


policy = np.zeros([capacity]*2)
value = np.zeros([capacity]*2)

# policy evaluation

# iterate through all states:
for ii in range(capacity):
    for jj in range(capacity):
        pass


def transition_probability(s0, a):
    ''' Transition probability from state s0 to other states with action a. '''
    pass


def expected_rental_reward(s, mu):
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

def expected_rental_reward2(s, mu):
    requests = np.asarray(range(s+1))
    rewards = 10 * np.asarray(range(s+1))
    p = poisson.pmf(requests, mu)

    # for number of requests equal to or larger than state:
    p[-1] = 1. - poisson.cdf(s-1, mu)

    return sum(rewards * p)
    

reward = [expected_rental_reward(s, 4) for s in range(20)]
reward2 = [expected_rental_reward2(s, 4) for s in range(20)]

print(reward)
print(reward2)


