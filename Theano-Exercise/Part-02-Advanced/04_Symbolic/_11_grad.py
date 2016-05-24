# Fill in the TODOs in this exercise, then run python _11_grad.py to see
# if your solution works!
from theano import tensor


def grad_sum(x, y, z):
    """
    Return dz/dx + dz/dy.
    :param x: a theano variable.
    :param y: a theano variable.
    :param z: a theano expression involving x and y.
    :return: dz/dx + dz/dy.
    """