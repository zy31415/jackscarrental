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
    p_ret = Poisson.pmf_series(ret, 20 - s)
    p = np.outer(p_req, p_ret)

    transp = np.asarray([p.trace(offset) for offset in range(-s, 20-s+1)])

    if action > 0:
        tail = sum(transp[-action:])
        transp[-action-1] += tail
        transp[-action:] = 0
    elif action < 0:
        tail = sum(transp[:-action])
        transp[-action] += tail
        transp[:-action] = 0

    return np.roll(transp, shift=action)


if __name__ == '__main__':
    state = np.arange(21)
    p = transition_probabilty(16, 3, 3, 10)

    print(sum(p))

    from matplotlib.pylab import plt

    plt.plot(p)
    plt.show()
