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
    Returns the elementwise product of a and b.
    :param a: a theano matrix
    :param b: a theano matrix
    :return: the elementwise product of a and b.
    """