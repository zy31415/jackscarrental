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


def bellman(action, s1, s2, value, reward1, reward2):
    transp1 = transition_probabilty(s1, request_mean_G1, return_mean_G1, -action)
    transp2 = transition_probabilty(s2, request_mean_G2, return_mean_G2, action)
    transp = np.outer(transp1, transp2)
    return reward1[s1] + reward2[s2] - 2 * action + discount * sum((transp * value).flat)


# policy evaluation
def policy_evaluation(policy, value, reward1, reward2):
    # TODO: Action cost is not deterministic, it's also stochastic. Correct it!

    global discount

    while True:
        diff = 0
        it = np.nditer([policy], flags=['multi_index'])
        while not it.finished:
            action = it[0]
            s1, s2 = it.multi_index

            _temp = value[s1, s2]

            value[s1, s2] = bellman(action, s1, s2, value, reward1, reward2)

            diff = max(diff, abs(value[s1, s2] - _temp))

            it.iternext()

        print(diff)
        if diff < .01:
            break


def policy_update(policy, value):

    it = np.nditer([policy], flags=['multi_index'])

    while not it.finished:
        s1, s2 = it.multi_index

        policy[s1, s2] = np.argmax([bellman(action, s1, s2, value, reward1, reward2) for action in range(6)])

        it.iternext()


class DPSolver(object):

    capacity = 20
    rental_reward = 10.

    request_mean_G1 = 3
    request_mean_G2 = 4
    return_mean_G1 = 3
    return_mean_G2 = 2

    discount = 0.9

    policy = None
    value = None

    def __init__(self):
        self.policy = np.zeros([self.capacity + 1]*2, int)
        self.value = np.zeros([self.capacity + 1]*2)

        ExpectedRentalReward.RENTAL_REWARD = self.rental_reward
        ExpectedRentalReward.CAPACITY = self.capacity

        self.reward1 = ExpectedRentalReward.get(self.request_mean_G1)
        self.reward2 = ExpectedRentalReward.get(self.request_mean_G2)

    def bellman(self, action, s1, s2):
        transp1 = transition_probabilty(s1, self.request_mean_G1, self.return_mean_G1, -action)
        transp2 = transition_probabilty(s2, self.request_mean_G2, self.return_mean_G2, action)
        transp = np.outer(transp1, transp2)

        return self.reward1[s1] + self.reward2[s2] - 2 * action + self.discount * sum((transp * self.value).flat)

    # policy evaluation
    def policy_evaluation(self):
        ''' Keep pocliy fixed and update value. '''
        # TODO: Action cost is not deterministic, it's also stochastic. Correct it!
        while True:
            diff = 0.
            it = np.nditer([self.policy], flags=['multi_index'])

            while not it.finished:
                action = it[0]
                s1, s2 = it.multi_index

                _temp = self.value[s1, s2]

                self.value[s1, s2] = self.bellman(action, s1, s2)

                diff = max(diff, abs(self.value[s1, s2] - _temp))

                it.iternext()

            print(diff)
            if diff < .01:
                break

    def policy_update(self):
        it = np.nditer([self.policy], flags=['multi_index'])
        while not it.finished:
            s1, s2 = it.multi_index
            self.policy[s1, s2] = np.argmax([self.bellman(action, s1, s2) for action in range(6)])
            it.iternext()



if __name__ == '__main__':

    solver = DPSolver()

    solver.policy_evaluation()

    for ii in range(4):
        solver.policy_update()

    import matplotlib.pylab as plt

    CS = plt.contour(solver.policy, levels=[0, 1, 2, 3, 4, 5])
    plt.clabel(CS)
    plt.show()

