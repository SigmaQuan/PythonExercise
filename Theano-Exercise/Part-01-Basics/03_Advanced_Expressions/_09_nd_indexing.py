# Fill in the TODOs in this exercise, then run the script to see if your
# solution works.
import numpy as np
from theano import config
from theano import shared
from theano import tensor


def shrink_tensor(x, w):
    """
    Return a theano TensorType variable containing all but the borders of
    x, i.e., discard the first and last w elements along each axis of x.
    Example:
        x = [0, 1, 2, 3, 4], w = 2 -> y = [2]
        x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], w = 1 -> y = [[5]]
    :param x: a theano TensorType variable.
    :param w: a theano integer scalar.
    :return: a theano TensorType variable containing all but the borders of
    x, i.e., discard the first and last w elements along each axis of x.
    """
    # print x.ndim
    return x[tuple([slice(w, -w)] * x.ndim)]

if __name__ == "__main__":
    # x = tensor.TensorType(config.floatX, (False, False, False))()
    x = tensor.TensorType(dtype=config.floatX, broadcastable=tuple([False]*3))()
    xv = np.random.randn(10, 11, 12).astype(config.floatX)
    y = shrink_tensor(x, shared(3)).eval({x:xv})
    assert y.shape == (4, 5, 6), y.shape
    for i in xrange(4):
        for j in xrange(5):
            for k in xrange(6):
                assert y[i, j, k] == xv[i + 3, j + 3, k + 3]

    print("SUCCESS!")
