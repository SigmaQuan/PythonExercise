# Fill in the TODOs in this exercise, then run python _04_function.py to see
# if your solution works!
import theano
import theano.tensor as tensor
from theano import function


def evaluate(x, y, expr, x_value, y_value):
    """
    Return the value of expr when x_value is substituted for x and y_value
    is substituted for y.
    :param x: a theano variable.
    :param y: a theano variable.
    :param expr: a theano expression involving x and y.
    :param x_value: a numpy value.
    :param y_value: a numpy value.
    :return: the value of expr when x_value is substituted for x and y_value
    is substituted for y.
    """
    f = function(inputs=[x, y], outputs=expr)
    return f(x_value, y_value)
    # return function([x, y], expr)(x_value, y_value)


if __name__ == "__main__":
    x = tensor.iscalar()
    y = tensor.iscalar()
    z = x + y
    assert evaluate(x, y, z, 1, 2) == 3
    print("SUCCESS!")
