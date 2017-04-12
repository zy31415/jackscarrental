import numpy as np
from scipy.stats import poisson


def poisson_cutoff(lam, cutoff):
    seq = np.arange(cutoff+1)
    p = poisson.pmf(seq, lam)
    # for number of requests equal to or larger than state:
    p[-1] = 1. - poisson.cdf(cutoff - 1, lam)

    return p


def transition_probabilty(s, req, ret, action=0):
    '''
    
    :param s: Current State
    :param req: Mean value of requests
    :param ret: Mean value of returns
    :param action: Action. Positive means move in. Negativ means move out.
    :return: Transition probability.
    '''
    p_req = poisson_cutoff(req, s)
    p_ret = poisson_cutoff(ret, 20 - s)
    p = np.outer(p_req, p_ret)

    transp = np.zeros(21)
    for nth, offset in enumerate(range(-s, 20-s+1), start=action):
        _trace = np.trace(p, offset)
        if 0 <= nth < 21:
            transp[nth] += _trace
        elif nth >= 21:
            transp[-1] += _trace
        elif nth < 0:
            transp[0] += _trace
        else:
            raise ValueError("Should not be here")

    return transp


if __name__ == '__main__':
    state = np.arange(21)
    p = transition_probabilty(16, 3, 3, -10)

    print(sum(p))

    from matplotlib.pylab import plt

    plt.plot(p)
    plt.show()
