# Fill in the TODOs in this exercise, then run the script to see if your
# solution works.
import numpy as np
import theano.tensor as T

def increment_odd(x):
    """
    Returns a theano vector equal to x, but with all odd-numbered elements
    incremented by 1.
    :param x: a theano vector.
    :return: a theano vector equal to x, but with all odd-numbered elements
    incremented by 1.
    """
    # y = x
    # for i in range(len(x)) and i % 2 == 0:
    #     y[i] += 1
    y = T.inc_subtensor(x[1::2], 1.)
    return y


if __name__ == "__main__":
    x = T.vector()
    xv = np.zeros((4,), dtype=x.dtype)
    yv = increment_odd(x).eval({x:xv})
    print yv
    assert np.allclose(yv, np.array([0., 1., 0., 1.]))
    print("SUCCESS!")
