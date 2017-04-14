import numpy as np
from poisson import Poisson


def transition_probabilty(s, req, ret, action=0):
    '''    
    :param s: Current State
    :param req: Mean value of requests
    :param ret: Mean value of returns
    :param action: Action. Positive means move in. Negativ means move out.
    :return: Transition probability.
    '''
    p_req = Poisson.pmf_series(req, s)
    p_ret = Poisson.pmf_series(ret, 25)
    p = np.outer(p_req, p_ret)

    transp = np.asarray([p.trace(offset) for offset in range(-s, 26)])

    # TODO: not hard code 5
    assert abs(action) <= 5, "action can be large than 5."

    if action == 0:
        transp[20] += sum(transp[21:])
        return transp[:21]

    if action > 0:
        transp[20-action] += sum(transp[20-action+1:])
        transp[20-action+1:] = 0

        return np.roll(transp, shift=action)[:20+1]

    action = -action
    transp[action] += sum(transp[:action])
    transp[:action] = 0

    transp[action+20] += sum(transp[action+20+1:])
    transp[action+20+1:] = 0

    return np.roll(transp, shift=-action)[:20+1]


if __name__ == '__main__':
    state = np.arange(21)

    m = 2

    p = transition_probabilty(m, 3, 4, 0)
    print(sum(p))

    p1 = transition_probabilty(m, 3, 4, 4)
    print(sum(p1))

    p2 = transition_probabilty(m, 3, 4, -4)
    print(sum(p2))

    from matplotlib.pylab import plt

    plt.plot(p, 'o-')
    plt.plot(p1, 'o-')
    plt.plot(p2, 'o-')
    plt.grid('on')
    plt.xticks(range(21))
    plt.show()
