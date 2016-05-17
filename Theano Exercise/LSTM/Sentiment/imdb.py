"""
Get tweet sentimet dataset.
"""
from __future__ import print_function
from six.moves import xrange
import six.moves.cPickle as pickle

import gzip
import os

import numpy
import theano


def prepare_data(seqs, labels, maxlen=None):
    """
    Create the matrices from the datasets.
    This pad each sequence to the same length: the length of the longuest
    sequence or maxlen. If maxlen is set, we will cut all sequence to this
    maximum length.
    This swap the axis!
    :param seqs: a set of sequence with the length size equal to maxlen.
    :param labels: a set of label.
    :param maxlen: the max length of a sequence
    :return: the sample set.
    """
    # x: a list of sentence
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_lables = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqsl.apppend(s)
                new_lables.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_lables
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels


def get_dataset_file(dataset, default_dataset, origin):
    """
    Look for it sa if it was a full path, if not, try local file, if not
    try in the data directory.
    Download dataset if it is not present.
    :param dataset: the data file path.
    :param default_dataset: the default data file name.
    :param origin: the address for down load data file.
    :return: the dataset.
    """
    # parsing the file path
    data_dir, data_file = os.path.split(dataset)
    # Check if dataset is in the data directory.
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == default_dataset:
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == default_dataset:
        from six.moves import urllib
        print('Download data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    return dataset


def load_data(
        path='imbd.pkl',
        n_words=100000,
        valid_portion=0.1,
        maxlen=None,
        sort_by_len=True):
    """
    Loads the move common sentiment analysis dataset.
    :param path: the path to the dataset (hel IMDB)
    :param n_words: the number of word to keep in the vocabulary. All extra
            words are set to unknow (1).
    :param valid_portion: the proportion of the full train set set used
            for the validation set.
    :param maxlen: the max sequence length used in the train/valid set.
    :param sort_by_len:Sort by the sequence length for the train, valid and
            test set. This allow faster execution as it cause less padding
            per mini-batch. Another mechanism must be used to shuffle the
            train set at each epoch.
    :return: train set, valid set, and test set.
    """
    # Load the dataset
    path = get_dataset_file(
        path,
        "imdb.pkl",
        "http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl"
    )

    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    train_set = pickle.load(f)
    test_set = pickle.load(f)
    f.close()
    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

    # split training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples*(1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)


    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)


    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)

    return train, valid, test
