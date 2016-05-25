import numpy as np
from theano import config
from theano import function
from theano import shared
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.compat.python2x import OrderedDict

num_vis = 2

class SimpleMLP(object):
    """
    An MLP with one sigmoid hidden layer and one linear output layer (for
    solving regression problems).
    """
    def __init__(self):
        rng = np.random.RandomState([1, 2, 3])
        self.num_hid = 3
        self.W_hid = shared(rng.randn(num_vis, self.num_hid).astype(
            config.floatX))
        self.W_out = shared(rng.randn(self.num_hid).astype(config.floatX))

    def fprop(self, X):
        """
        The forward propagation of MLP.
        :param X: a theano matrix of input samples: each row is an example;
        each column is a feature.
        :return:
        H: a theano matrix of hidden unit values.
        y_hat: a theano vector of outputs. Output i is the predicted value
        for example i.
        """
        H = tensor.nnet.sigmoid(tensor.dot(X, self.W_hid))
        y_hat = tensor.dot(H, self.W_out)

        return H, y_hat

def loss(y_hat, y):
    """
    Return a expression for the loss on one mini-batch training.
    :param y_hat: a mini-batch of predictions.
    :param y: a mini-batch of targets.
    :return: the expression for the loss on one mini-batch training.
    """
    return tensor.sqr(y - y_hat).mean()

def two_step_backprop(mlp):
    """
    Back-propagation of MLPs.
    :param mlp: a SimpleMLP instance.
    :return: two theano functions.
    f1: a theano function,
        takes two arguments:
            1) a mini-batch of examples and
            2) a mini-batch of targets.
        return two values:
            1) the gradient of the loss on mlp.W_out;
            2) an auxiliary value of your choosing.
    f2: a theano function,
        takes two arguments:
            1) a mini-batch of example and
            2) the auxiliary value returned by f1.
        return:
            the gradient of the loss on mlp.W_hid.
        Should not make use of mlp.W_out at all!
    """
    X = tensor.matrix()
    y = tensor.vector()
    H, y_hat = mlp.fprop(X)
    # grad_W_out = tensor.grad(self.loss(y_hat, y), self.W_out,
    #                          consider_constant=[X, y, self.W_hid])
    # h_bias = tensor.scalar()
    g_W_out, g_H = tensor.grad(loss(y_hat, y), [mlp.W_out, H])
    f1 = function([X, y], [g_W_out, g_H])

    known_grads = OrderedDict()
    known_grads[H] = g_H

    g_W_hid = tensor.grad(None, mlp.W_hid, known_grads=known_grads)
    f2 = function([X, g_H], g_W_hid)

    return f1, f2


if __name__ == "__main__":
    mlp = SimpleMLP()
    X = tensor.matrix()
    y = tensor.vector()
    H, y_hat = mlp.fprop(X)
    l = loss(y_hat, y)
    # g_W, g_w = tensor.grad(cost=l, wrt=[mlp.W_hid, mlp.W_out])
    g_W, g_w = tensor.grad(cost=l, wrt=[mlp.W_hid, mlp.W_out])
    rng = np.random.RandomState([1, 2, 3])
    m = 5
    f = function(inputs=[X, y], outputs=[g_W, g_w])
    X = rng.randn(m, num_vis).astype(X.dtype)
    y = rng.randn(m).astype(y.dtype)
    g_W, g_w = f(X, y)
    f1, f2 = two_step_backprop(mlp)
    g_w2, aux = f1(X, y)
    assert np.allclose(g_w, g_w2)
    # Give w_out the wrong size to make sure f2 can't use it
    mlp.W_out.set_value(np.ones(1).astype(mlp.w_out.dtype))
    g_W2 = f2(X, aux)
    print('SUCCESS!')
