# Fill in the TODOs in this exercise, then run python _02_vector.py to see
# if your solution works!
import numpy as np
from theano import function
import theano.tensor as tensor
# raise NotImplementedError("TODO: add any other imports you need")

def make_vector():
    """
    Return a new theano vector.
    :return:
    """
    return tensor.vector()


def make_matrix():
    """
    Return a new theano matrix.
    :return:
    """
    return tensor.matrix()


def elemwise_mul(a, b):
    """
    Returns the element-wise product of a and b.
    :param a: a theano matrix
    :param b: a theano matrix
    :return: the element-wise product of a and b.
    """
    c = a*b
    return c


def matrix_vector_mul(a, b):
    """
    Return the matrix-vector product of a and b.
    :param a: a theano matrix.
    :param b: a theano vector.
    :return: the matrix-vector product of a and b.
    """
    return a*b


if __name__ == "__main__":
    a = make_vector()
    b = make_vector()
    c = elemwise_mul(a, b)
    d = make_matrix()
    e = matrix_vector_mul(d, c)

    f = function([a, b, d], e)

    rng = np.random.RandomState([1, 2, 3])
    a_value = rng.randn(5).astype(a.dtype)
    b_value = rng.randn(5).astype(b.dtype)
    c_value = a_value * b_value
    d_value = rng.randn(5, 5).astype(d.dtype)
    expected = np.dot(d_value, c_value)

    actual = f([a_value, b_value, d_value], expected)

    assert np.allclose(actual, expected)
    print("SUCCESS!")

