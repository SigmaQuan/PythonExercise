# Fill in the TODOs in this exercise, then run python _03_tensor.py to see
# if your solution works!
import numpy as np
from theano import function
import theano.tensor as tensor
# raise NotImplementedError("TODO: add any other imports you need")

def make_tensor(dim):
    """
    Returns a new theano tensor with no broadcastable dimensions.
    :param dim: the total number of dimensions of the tensor.
    :return: a new theano tensor with no broadcastable dimensions.
    """
    # return tensor.TensorType('float64', (False,)*dim)
    return tensor.TensorType(broadcastable=tuple([False] * dim), dtype='float32')()
    # raise NotImplementatedError("TODO: implement this function")


def broadcasted_add(a, b):
    """
    Returns c, a 4D theano tensor, where
    c[i,j,k,l] = a[l,k,i]+b[i,j,k,l] for all i, j, k, l
    :param a: a 3D theano tensor
    :param b: a 4D theano tensor
    :return:
    """
    # b0, b1, b2, b3 = tensor.shape(b)
    # c = np.zeros((b0, b1, b2, b3))
    # for i in range(b0):
    #     for j in range(b1):
    #         for k in range(b2):
    #             for l in range(b3):
    #                 c[i, j, k, l] = a[l, k, i] + b[i, j, k, l]

    return a.dimshuffle(2, 'x', 1, 0) + b
    # raise NotImplementatedError("TODO: implement this function")


def partial_max(a):
    """
    Returns b, a theano matrix, where
    b[i,j] = max_{k,l} a[i,k,l,j] for all i,j
    :param a:
    :return: a theano matrix
    """
    return tensor.max(a, axis=[1, 2])
    # return a.max(axis=(1, 2))
    # raise NotImplementatedError("TODO: implement this function")


if __name__ == "__main__":
    a = make_tensor(3)
    b = make_tensor(4)
    c = broadcasted_add(a, b)
    d = partial_max(c)

    f = function([a, b, ], d)

    rng = np.random.RandomState([1, 2, 3])
    a_value = rng.randn(2, 2, 2).astype(a.dtype)
    b_value = rng.rand(2, 2, 2, 2).astype(b.dtype)
    c_value = np.transpose(a_value, (2, 1, 0))[:, None, :, :] + b_value
    expected = c_value.max(axis=1).max(axis=1)

    actual = f(a_value, b_value)

    assert np.allclose(actual, expected), (actual, expected)
    print("SUCCESS!")
