# Fill in the TODOs and run python _12_traverse.py to see if your solution
# works!
import numpy as np
from theano import tensor
from theano.gof import Variable


def arg_to_softmax(prob):
    """
    Oh no! Some has passed you probability output, "prob", of a softmax
    function, and you want the unnormalized log probabilty -- the argument
    to the softmax.
    Verify that prob really is the output of a softmax. Raise a TypeError
    if it is not.
    If it is, return the argument to the softmax.
    :param prob:
    :return:
    """
    #***** NOT UNDERSTAND
    if not isinstance(prob, Variable):
        raise TypeError()

    if prob.owner is None:
        raise  TypeError()

    owner = prob.owner

    if not isinstance(owner.op, tensor.nnet.Softmax):
        raise TypeError()

    rval, = owner.inputs
    #***** NOT UNDERSTAND

    return rval


if __name__ == "__main__":
    x = np.ones((5, 4))
    try:
        arg_to_softmax(x)
        raise Exception("You should have raise an error.")
    except TypeError:
        pass

    x = tensor.matrix()
    try:
        arg_to_softmax(x)
        raise Exception("You should have raise an error.")
    except TypeError:
        pass

    y = tensor.nnet.sigmoid(x)
    try:
        arg_to_softmax(y)
        raise Exception("You should have raise an error.")
    except TypeError:
        pass

    y = tensor.nnet.softmax(x)
    rval = arg_to_softmax(y)
    assert rval is x

    print("SUCCESS!")