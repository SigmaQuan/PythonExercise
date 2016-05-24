import numpy as np
from theano import config
from theano import shared
# from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams


def bernoulli_samples(p):
    """
    Returns a sampled variable values v, with each element of v being 1 with
    the probability given by the corresponding element of p.
    :param p: a theano variable with elements in the interval [0, 1].
    :return:
    """
    # srng = RandomStreams(seed=234)
    # bernoulli = [srng.binomial(p) for i in xrange(len(p))]
    rng_factory = MRG_RandomStreams(42)
    bernoulli = rng_factory.binomial(p=p, size=p.shape, dtype=p.dtype, n=1)
    return bernoulli


if __name__ == "__main__":
    p = shared(np.array(range(11), dtype=config.floatX)/10)
    # print p
    s = bernoulli_samples(p).reshape((1, 11))
    m = 100
    samples = np.concatenate([s.eval() for i in xrange(m)], axis=0)
    zeros = samples == 0
    ones = samples == 1
    combined = zeros + ones
    assert combined.min() == 1.
    assert combined.max() == 1.
    mean = samples.mean(axis=0)
    # Unlikely your mean would be off by this much (but possible)
    assert np.abs(mean - p.get_value()).max() < 0.2
    print("SUCCESS!")
