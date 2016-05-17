"""
Build a tweet sentiment analyzer
"""
from __future__ import print_function

import numpy
import six.moves.cPickle as pickle
import sys
import theano
import theano.tensor as tensor
import time
from collections import OrderedDict
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import imdb

# help(imdb.prepare_data)

datasets = {'imdb':(imdb.load_data, imdb.prepare_data)}

# Set the random number generator's seeds for consistency
SEED = 123
numpy.random.seed(SEED)


def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_mimibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    :param n:
    :param minibatch_size:
    :param shuffle:
    :return:
    """


def get_dataset(name):
    return datasets(name[0], datasets[name][1])


def zipp(params, tparams):
    """
    When the model are reload. Needed for the GPU stuff.
    :param params:
    :param tparams:
    :return:
    """


def unzip(zipped):
    """
    When pickle the model. Need for the GPU stuff.
    :param zipped:
    :return:
    """


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(
        use_noise,
            (state_before *
             trng.binomial(state_before.shape,
                           p=0.5,
                           n=1,
                           dtype=state_before.dtype)),
        state_before*0.5
    )
    return proj


