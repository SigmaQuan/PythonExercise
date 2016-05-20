# Fill in the TODOs in this exercise, then run python_01_scalar.py to see
# if your solution works!
import numpy as np
from theano import function
import theano.tensor as tensor
# raise NotImplementedError("TODO: add any other imports you need")

def make_scalar():
    """
    Return a new Theano scalar.
    :return: a new Theano scalar.
    """
    # raise NotImplementedError("TODO: implement this function.")
    return tensor.dscalar()


def log(x):
    """
    Returns the logarithm of a Theano scalar x.
    :param x:
    :return:
    """
    return tensor.log(x)


def add(x, y):
    """
    Adds two theano scalars together and returns the result.
    :param x:
    :param y:
    :return:
    """
    return x + y


if __name__ == '__main__':
    a = make_scalar()
    b = make_scalar()
    c = log(b)
    d = add(a, c)
    f = function([a, b], d)
    a = np.cast[a.dtype](1.)
    b = np.cast[b.dtype](2.)
    actual = f(a, b)
    expected = 1. + np.log(2.)
    print actual
    print expected
    assert np.allclose(actual, expected)
    print "SUCCESS!"
