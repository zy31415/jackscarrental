import numpy as np

from expected_rental_reward import ExpectedRentalReward
from transition_probabilty import transition_probabilty

# Problem definition:
capacity = 20
discount = .9
rental_reward = 10.
transfer_reward = -2.

request_mean_G1 = 3
return_mean_G1 = 3
request_mean_G2 = 4
return_mean_G2 = 2


# policy evaluation

def policy_evaluation(policy, value, reward1, reward2):
    global discount

    while True:
        diff = 0
        it = np.nditer([policy], flags=['multi_index'])
        while not it.finished:
            action = it[0]
            s1, s2 = it.multi_index

            _temp = value[s1, s2]

            transp1 = transition_probabilty(s1, request_mean_G1, return_mean_G1, -action)
            transp2 = transition_probabilty(s2, request_mean_G2, return_mean_G2, action)
            transp = np.outer(transp1, transp2)

            value[s1, s2] = reward1[s1] + reward2[s2] - 2 * action + discount * sum((transp * value).flat)

            diff = max(diff, abs(value[s1, s2] - _temp))

            it.iternext()

        print(diff)
        if diff < .01:
            break


def array_index_to_diagonal_index(s1, s2):
    nth_diag = s2 + s1
    nth_ele = s2 if nth_diag < 20 else 20 - s1
    return nth_diag, nth_ele


def policy_update(policy, value):

    max_index = [np.argmax(np.flipud(value).diagonal(ii)) for ii in range(-20, 21)]

    it = np.nditer([policy], flags=['multi_index'])

    while not it.finished:
        s1, s2 = it.multi_index

        nth_diag, nth_ele = array_index_to_diagonal_index(s1, s2)

        _dis = max_index[nth_diag] - nth_ele

        policy[s1, s2] = min(_dis, 5) if _dis >= 0 else max(_dis, -5)

        it.iternext()


if __name__ == '__main__':

    reward1 = ExpectedRentalReward.get(request_mean_G1)
    reward2 = ExpectedRentalReward.get(request_mean_G2)

    # Policy to be evaluated
    policy = np.zeros([capacity + 1]*2, int)

    # Initial value
    value = np.zeros([capacity + 1]*2)

    for ii in range(4):
        policy_evaluation(policy, value, reward1, reward2)
        policy_update(policy, value)

    import matplotlib.pylab as plt

    CS = plt.contour(policy, levels=[0, 1, 2, 3, 4, 5])
    plt.clabel(CS)
    plt.show()

