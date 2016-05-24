# Fill in the TODOs in this exercise, then run python _09_tensor.py to see
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
    raise NotImplementatedError("TODO: implement this function")


def broadcasted_add(a, b):
    """
    Returns c, a 4D theano tensor, where
    c[i,j,k,l] = a[l,k,i]+b[i,j,k,l]
    for all i, j, k, l
    :param a: a 3D theano tensor
    :param b: a 4D theano tensor
    :return:
    """
    raise NotImplementatedError("TODO: implement this function")