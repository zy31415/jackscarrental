import numpy as np

from poisson import Poisson


class DPSolver(object):

    capacity = 20
    rental_reward = 10.
    moving_cost = 2.
    max_moving = 5

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

        self._reward1 = self.expected_rental_reward(self.request_mean_G1)
        self._reward2 = self.expected_rental_reward(self.request_mean_G2)

    def bellman(self, action, s1, s2):
        transp1 = self.transition_probabilty(s1, self.request_mean_G1, self.return_mean_G1, -action)
        transp2 = self.transition_probabilty(s2, self.request_mean_G2, self.return_mean_G2, action)
        transp = np.outer(transp1, transp2)

        return self._reward1[s1] + self._reward2[s2] - self.expected_moving_cost(s1, s2, action) + \
               self.discount * sum((transp * self.value).flat)

    # policy evaluation
    def policy_evaluation(self):
        ''' Keep pocliy fixed and update value. '''
        while True:
            diff = 0.
            it = np.nditer([self.policy], flags=['multi_index'])

            while not it.finished:
                action = it[0]
                s1, s2 = it.multi_index

                _temp = self.value[s1, s2]

                self.value[s1, s2] = self.bellman(action=action, s1=s1, s2=s2)

                diff = max(diff, abs(self.value[s1, s2] - _temp))

                it.iternext()

            print(diff)
            if diff < .01:
                break

    def policy_update(self):
        it = np.nditer([self.policy], flags=['multi_index'])
        while not it.finished:
            s1, s2 = it.multi_index

            _max_val = -1
            _pol = None

            for act in range(-self.max_moving, self.max_moving + 1):
                _val = self.bellman(action=act, s1=s1, s2=s2)
                if _val > _max_val:
                    _pol = act

            self.policy[s1, s2] = _pol

            it.iternext()

    def expected_moving_cost(self, s1, s2, action):
        if action == 0:
            return 0.

        # moving from s1 into s2
        if action > 0:
            p = self.transition_probabilty(s1, self.request_mean_G1, self.return_mean_G1)
            cost = np.asarray(
                [ii if ii < action else action for ii in range(self.capacity+1)]
            ) * self.moving_cost

            return cost.dot(p)

        # moving from s2 into s1
        p = self.transition_probabilty(s2, self.request_mean_G2, self.return_mean_G2)
        cost = np.asarray(
                [ii if ii < -action else -action for ii in range(self.capacity+1)]
            ) * self.moving_cost

        return cost.dot(p)

    @classmethod
    def expected_rental_reward(cls, expected_request):
        return np.asarray([cls._state_reward(s, expected_request) for s in range(cls.capacity + 1)])

    @classmethod
    def _state_reward(cls, s, mu):
        rewards = cls.rental_reward * np.arange(s + 1)
        p = Poisson.pmf_series(mu, cutoff=s)
        return rewards.dot(p)

    def transition_probabilty(self, s, req, ret, action=0):
        '''    
        :param s: Current State
        :param req: Mean value of requests
        :param ret: Mean value of returns
        :param action: Action. Positive means move in. Negativ means move out.
        :return: Transition probability.
        '''

        _ret_sz = self.max_moving + self.capacity

        p_req = Poisson.pmf_series(req, s)
        p_ret = Poisson.pmf_series(ret, _ret_sz)
        p = np.outer(p_req, p_ret)

        transp = np.asarray([p.trace(offset) for offset in range(-s, _ret_sz + 1)])

        # TODO: not hard code 5
        assert abs(action) <= self.max_moving, "action can be large than %s." % self.max_moving

        if action == 0:
            transp[20] += sum(transp[21:])
            return transp[:21]

        if action > 0:
            transp[self.capacity-action] += sum(transp[self.capacity-action+1:])
            transp[self.capacity-action+1:] = 0

            return np.roll(transp, shift=action)[:self.capacity+1]

        action = -action
        transp[action] += sum(transp[:action])
        transp[:action] = 0

        transp[action+self.capacity] += sum(transp[action+self.capacity+1:])
        transp[action+self.capacity+1:] = 0

        return np.roll(transp, shift=-action)[:self.capacity+1]


if __name__ == '__main__':

    solver = DPSolver()

    for ii in range(4):
        solver.policy_evaluation()
        solver.policy_update()

    print(solver.policy)

    import matplotlib.pylab as plt

    plt.subplot(121)
    CS = plt.contour(solver.policy, levels=range(-6, 6))
    plt.clabel(CS)
    plt.xlim([0, 20])
    plt.ylim([0, 20])
    plt.axis('equal')
    plt.xticks(range(21))
    plt.yticks(range(21))
    plt.grid('on')

    plt.subplot(122)
    plt.pcolor(solver.value)
    plt.colorbar()
    plt.axis('equal')

    plt.show()

