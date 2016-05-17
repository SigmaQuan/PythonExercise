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
    idx_list=numpy.arange(n, dtype='int32')
    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatchs =[]
    minibatch_start = 0
    for i in range(n//minibatch_size):  #//: Floor
        minibatchs.append(
            idx_list[minibatch_start: minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    # Make a minibatch out of what is left.
    if(minibatch_start != n):
        minibatchs.append(idx_list[minibatch_start:])

    return zip(range(len(minibatchs)), minibatchs)


def get_dataset(name):
    return datasets(name[0], datasets[name][1])


def zipp(params, tparams):
    """
    When the model are reload. Needed for the GPU stuff.
    :param params: parameter of model.
    :param tparams: make the parameter to zipp.
    :return: none.
    """
    for kk, vv in params.items():
        tparams[kk].setvalue(vv)


def unzip(zipped):
    """
    When pickle the model. Need for the GPU stuff.
    :param zipped:
    :return: parameter of model.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    """
    Dropout some nodes in a layer.
    :param state_before: the state before dropout.
    :param use_noise: the noise for..
    :param trng: for producing binomial distribution.
    :return: the switch vector.
    """
    projection = tensor.switch(
        use_noise,
        (state_before *
         trng.binomial(state_before.shape,
                       p=0.5,
                       n=1,
                       dtype=state_before.dtype)),
        state_before*0.5
    )
    return projection


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    """
    Global (not LSTM) parameter. For the embedding and the classifier.
    :param options:
    :return:
    """
    params = OrderedDict()
    # embedding
    randn = numpy.random.rand(options['n_word'], options['dim_proj'])
    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params = get_layer(options['encoder'])[0](
        options, params, prefix=options['encoder'])

    # classifier
    params['U'] = 0.01 * numpy.random.randn(
        options['dim_proj'], options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params


def load_params(path, params):


def init_tparams(params):


def get_layer(name):


def ortho_weigh(ndim):


def parma_init_lstm(options, params, prefix='lstm'):


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):


# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
layers={'lstm': (param_init_lstm, lstm_layer)}


def sgd(lr, tparams, grads, x, mask, y, cost):
    """
    Stochastic Gradient Descent
    note: A more complicated of sgd then needed. This is done like that for
    adadelta and rmsprop.
    :param lr:
    :param tparams:
    :param grads:
    :param x:
    :param mask:
    :param y:
    :param cost:
    :return:
    """


def adadelta(lr, tparams, grads, x, mask, y, cost):
    """
    An adaptive learning rate optimizer.
    :param lr: Theano SharedVariable, initial learning rate.
    :param tparams: Theano SharedVariable, model parameters.
    :param grads: Theano variable, gradients of cost w.r.t. parameters.
    :param x: Theano variable, model inputs.
    :param mask: Theano variable, sequence mask.
    :param y: Theano variable, targets.
    :param cost: Theano variable, objective function to minimize.
    :return:
    Notes: for more information, see Matthew D. Zeiler, ADADELTA: An
        adaptive learning rate method, arXiv:1212.5601.
    """


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    """
    A variant of SGD that scales the step size by running average of the
    recent step norms.
    :param lr: Theano SharedVariable, initial learning rate.
    :param tparams: Theano SharedVariable, model parameters.
    :param grads: Theano variable, gradients of cost w.r.t to parameters.
    :param x: Theano varaible, model inputs.
    :param mask: Theano variable, sequence mask.
    :param y: Theano variable, targets.
    :param cost: Theano variable, objective function to minimize.
    :return:
    Note: for more information, see Geoff Hinton, Neural networks for
        machine learning, lecture 6a,
        htttp://cs.toronto.edu/~tijmen/csc321/sildes/lecture_slides_lec5.pdf
    """


def build_model(tparams, options):


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """
    If you want to to use a trained model, this is useful to compute the
    probablilities of new examples.
    :param f_pred_prob:
    :param prepare_data:
    :param data:
    :param iterator:
    :param verbose:
    :return:
    """


def pred_error(f_pred, prepare_data, data, iterator, verbose=False):


def train_lstm(
        dim_proj=128, patience=10, max_epochs=5000, dispFreq=10,
        decay_c=0., lrate=0.0001, n_words=10000, optimizer=adadelta,
        encoder='lstm', saveto='lstm_model.npz', validFreq=370,
        saveFreq=1110, maxlen=100, batch_size=16, valid_batch_size=64,
        dataset='imdb',
        # Parameters for extra option
        noise_std=0., use_dropout=True, reload_model=None, test_size=-1):
    """
    Training LSTM.
    :param dim_proj: word embeding dimension and  LSTM number of hidden
        units.
    :param patience: number of each to wait before early stop if no
        progress.
    :param max_epochs: the maximum number of epoch to run.
    :param dispFreq:display to stdout the training progress every N
        updates.
    :param decay_c: weight decay for the classifier applied to the U
        weights.
    :param lrate: learning rate for sgd (not used for adadelta and rmsprop)
    :param n_words: vocabulary size.
    :param optimizer: sgd, adadelt and rmsprop variable, sgd very hard to
        use, not recommended (probably need momentum and decaying learning
        rate).
    :param encoder: TODO: can be removed must be lstm.
    :param saveto: the best model will be saved there.
    :param validFreq: compute teh validation error after this number of
        update.
    :param saveFreq: save the parameters after every saveFreq updates.
    :param maxlen: sequence longer than this get ignored.
    :param batch_size: the batch size during training.
    :param valid_batch_size: the batch size during validation/test set.
    :param dataset: data file name.
    :param noise_std: the standard deviation of noise.
    :param use_dropout:if False slightly faster, but worst test error.
        This frequently need a bigger model.
    :param reload_model: path to a saved model we want to start from.
    :param test_size: if > 0, we keep only this number of test example.
    :return:
    """

if __name__=='__main__':
    # See function train for all possible parameter and there definition.
    train_lstm(max_epochs=100, test_size=500)