# Calculate the energy function of RBM, the gradient of the function.
import numpy as np
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams
import theano.tensor as tensor
from theano import function


def energy(W, V, H):
    """
    Calculate the energy function of RBM. This RBM has no biases, only
    weights.
    :param W: a theano matrix of RBM weights num_visible x num_hidden.
    :param V: a theano matrix of assignments to visible units: each row is
    another configuration; each column corresponds to a different unit.
    :param H: a theano matrix of assignments to hidden units: each row is
    another configuration; each column corresponds to a different unit.
    :return: a theano vector E, element i gives the energy of
    configuration (V[i, :], H[i, :]).
    """
    # return -tensor.dot(tensor.dot(V, W), H).sum(axis=1)
    return -(tensor.dot(V, W) * H).sum(axis=1)



def grad_expected_energy(W, V, H):
    """
    Calculate the derivatives of the expected gradient of the energy.
    :param W: a theano matrix of RBM weights num_visible x num_hidden.
    :param V: a theano matrix of assignments to visible units: each row is
    another configuration; each column corresponds to a different unit.
    :param H: a theano matrix of assignments to hidden units: each row is
    another configuration; each column corresponds to a different unit.
    :return: a theano matrix dW, element i,j are the gradient of W.
    """
    # return tensor.grad(energy(W, V, H), W)
    # return tensor.grad(energy(W, V, H).mean(), W, consider_constant=[V, H])
    return tensor.grad(cost=energy(W, V, H).mean(), wrt=W, consider_constant=[V, H])


if __name__ == "__main__":
    m = 2
    nv = 3
    nh = 4
    h0 = tensor.alloc(1., m, nh)
    rng_factory = MRG_RandomStreams(234)
    W = rng_factory.normal(size=(nv, nh), dtype=h0.dtype)
    pv = tensor.nnet.sigmoid(tensor.dot(h0, W.T))
    v = rng_factory.binomial(p=pv, size=pv.shape, dtype=W.dtype)
    ph = tensor.nnet.sigmoid(tensor.dot(v, W))
    h = rng_factory.binomial(p=ph, size=ph.shape, dtype=W.dtype)

    #***** NOT UNDERSTAND
    class _ElemwiseNoGradient(theano.tensor.Elemwise):
        def grad(self, inputs, output_gradients):
            raise TypeError("You shouldn't be differentiating "
                            "through the sampling process.")
            return [theano.gradient.DisconnectedType()()]
    block_gradient = _ElemwiseNoGradient(theano.scalar.identity)
    #***** NOT UNDERSTAND

    v = block_gradient(v)
    h = block_gradient(h)

    g = grad_expected_energy(W, v, h)
    stats = tensor.dot(v.T, h)/m
    f = function([], [g, stats])
    g, stats = f()
    assert np.allclose(g, -stats)
    print("SUCCESS!")
