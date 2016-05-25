import numpy as np
from theano import config
from theano import function
from theano import shared
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams


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

    def loss(self, y_hat, y):
        """
        Return a expression for the loss on one mini-batch training.
        :param y_hat: a mini-batch of predictions.
        :param y: a mini-batch of targets.
        :return: the expression for the loss on one mini-batch training.
        """
        return tensor.sqr(y, y_hat).mean()

    def two_step_backprop(self, mlp):
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
        H, y_hat = self.fprop(X)
        grad_W_out = tensor.grad(self.loss(y_hat, y), self.W_out,
                                 consider_constant=[X, y, self.W_hid])
        h_bias = tensor.scalar()
        f1 = function([X, y], [grad_W_out, h_bias])

        grad_W_hid = tensor.grad(self.loss(y_hat, y), self.W_hid)
        f2 = function([X, h_bias])

        return f1, f2