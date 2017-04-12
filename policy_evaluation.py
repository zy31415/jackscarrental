import numpy as np

from expected_rental_reward import ExpectedRentalReward
from transition_probabilty import transition_probabilty

# Problem definition:
capacity = 20
discount = 0.9
rental_reward = 10.
transfer_reward = -2.

request_mean_G1 = 3
return_mean_G1 = 3
request_mean_G2 = 4
return_mean_G2 = 2


# policy evaluation

def policy_evaluation(policy, value, reward):
    global discount

    while True:
        diff = 0
        it = np.nditer([policy, reward], flags=['multi_index'])
        while not it.finished:
            action, rwd = it[0], it[1]
            s1, s2 = it.multi_index

            _temp = value[s1, s2]

            transp1 = transition_probabilty(s1, request_mean_G1, return_mean_G1, -action)
            transp2 = transition_probabilty(s2, request_mean_G2, return_mean_G2, action)
            transp = transp1[:, np.newaxis] * transp2

            value[s1, s2] = rwd - 2 * action + discount * sum((transp * value).flat)

            diff = max(diff, abs(value[s1, s2] - _temp))

            it.iternext()

        print(diff)
        if diff < 0.01:
            break


if __name__ == '__main__':

    ExpectedRentalReward.set(request_mean_G1, request_mean_G2)

    # TODO: can be simplified, don't need a matrix.
    reward = ExpectedRentalReward.get()

    # Policy to be evaluated
    policy = np.zeros([capacity + 1]*2, int)

    # Initial value
    value = np.zeros([capacity + 1]*2)

    policy_evaluation(policy, value, reward)

    import matplotlib.pylab as plt

    plt.pcolor(value)
    plt.colorbar()
    plt.show()

