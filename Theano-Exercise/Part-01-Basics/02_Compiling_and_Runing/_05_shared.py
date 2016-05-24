# Fill in the TODOs in this exercise, then run python _05_shared.py to see
# if your solution works!
import numpy as np
from theano import shared
from theano import function
from theano.compat.python2x import OrderedDict


def make_shared(shape):
    """
    Returns a theano shared variable containing a tensor of the specified
    shape.
    :param shape:
    :return:
    """
    return shared(np.zeros(shape))


def exchange_shared(a, b):
    """
    Uses get_value and set_value to swap the values stored in a and b.
    :param a: a theano shared variable.
    :param b: a theano shared variable.
    :return: None.
    """
    # c = a.get_value()
    # a.set_value(b.get_value())
    # b.set_value(c)
    temp = a.get_value()
    a.set_value(b.get_value())
    b.set_value(temp)


def make_exchange_func(a, b):
    """
    Returns a theano function f, that, when called, swaps the values in a
    and b, f should not return anything.
    :param a: a theano shared variable.
    :param b: a theano shared variable.
    :return:
    """
    # wrong answer
    # x1 = make_shared(0)
    # x2 = make_shared(0)
    # return function(inputs=[x1, x2], updates=exchange_shared(x1, x2))(a, b)
    updates = OrderedDict()
    updates[a] = b
    updates[b] = a
    return function([], updates=updates)


if __name__ == "__main__":
    a = make_shared((5, 4, 3))
    assert a.get_value().shape == (5, 4, 3)

    b = make_shared((5, 4, 3))
    assert b.get_value().shape == (5, 4, 3)

    a.set_value(np.zeros((5, 4, 3), dtype=a.dtype))
    b.set_value(np.ones((5, 4, 3), dtype=b.dtype))
    exchange_shared(a, b)
    assert np.all(a.get_value() == 1.)
    assert np.all(b.get_value() == 0.)

    f = make_exchange_func(a, b)
    rval = f()
    assert isinstance(rval, list)
    assert len(rval) == 0
    assert np.all(a.get_value() == 0.)
    assert np.all(b.get_value() == 1.)

    print("SUCCESS!")
