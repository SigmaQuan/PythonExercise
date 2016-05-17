from __future__ import print_function
from six.moves import xrange
import six.moves.cPickle as pickle

import gzip
import os

import numpy
import theano

def prepare_data(seqs, labels, maxlen=None)
    """
    Create the matrices from the datasets.
    This pad each sequence to the same length: the length of the longuest
    sequence or maxlen. If maxlen is set, we will cut all sequence to this
    maximum length.
    This swap the axis!
    :param seqs: a set of sequence with the length size equal to maxlen.
    :param labels: a set of label.
    :param maxlen: the max length of a sequence
    :return:
    """