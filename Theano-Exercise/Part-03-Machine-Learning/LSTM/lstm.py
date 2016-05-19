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

datasets = {'imdb': (imdb.load_data, imdb.prepare_data)}

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
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weigh(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def parma_init_lstm(options, params, prefix='lstm'):
    """
    Initial the LSTM parameter.
    :param options:
    :param params:
    :param prefix:
    :return:
    """
    W = numpy.concatenate(
        [ortho_weigh(options['dim_proj']),
         ortho_weigh(options['dim_proj']),
         ortho_weigh(options['dim_proj']),
         ortho_weigh(options['dim_proj'])],
        axis=1
    )
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate(
        [ortho_weigh(options['dim_proj']),
         ortho_weigh(options['dim_proj']),
         ortho_weigh(options['dim_proj']),
         ortho_weigh(options['dim_proj'])],
        axis=1
    )
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4*options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim]
        return  _x[:, n*dim:(n+1)*dim]

    def _step(m_, x_, h_, c_):  #??
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f*c_ + i*c
        c = m_[:, None]*c + (1.-m_)[:, None]*c_

        h = o*tensor.tanh(c)
        h = m_[:, None]*h + (1.-m_)[:, None]*h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefx, 'W')])+
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    rval, upates =  theano.scan(
        _step,
        sequences=[mask, state_below],
        outputs_info=[tensor.alloc(numpy_floatX(0.), n_samples, dim_proj),
                      tensor.alloc(numpy_floatX(0.), n_samples, dim_proj)],
        name=_p(prefix, '_layers'),
        n_steps=nsteps
    )

    return rval[0]


# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
layers={'lstm': (param_init_lstm, lstm_layer)}


def sgd(lr, tparams, grads, x, mask, y, cost):
    """
    Stochastic Gradient Descent
    note: A more complicated of sgd then needed. This is done like that for
    adadelta and rmsprop.
    :param lr: Theano SharedVaraible, learning rate.
    :param tparams: Theano SharedVaraible, model parameters.
    :param grads: Theano variable, gradient of cost w.r.t. parameters.
    :param x: Theano variable, sequence mask.
    :param mask: Theano variable, sequence mask.
    :param y: Theano variable, targets.
    :param cost: Theano variable, objective function to minimize.
    :return:
    """
    # New set of shared variable that will contain the gradient for a
    # mini-batch.
    gshared = [theano.shared(p.get_value()*0., name='%s_grad'%k)
               for k, p in tparams.items()]
    gsup = [(gs,g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not updates
    # the weights.
    f_grd_shared = theano.function([x, mask, y], cost, updates=gsup,
                                   name = 'sgd_f_grad_shared')

    pup = [(p, p-lr*g) for p, g in zip(tparams.values(), gshared)]

    # Function that update the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup, name='sgd_f_update')

    return f_grd_shared, f_update


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
    zipped_grads = [theano.shared(p.get_value()*numpy_floatX(0.),
                                  name='%s_grad'%k)
                   for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value()*numpy_floatX(0.),
                                 name='%s-rup2'%k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value()*numpy_floatX(0.),
                                    name='%s_rgrad2'%k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95*rg2+0.05*(g**2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup+rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2+1e-6)/tensor.sqrt(rg2+1e-6)*zg
             for zg, ru2, rg2, in zip(zipped_grads,
                                      running_up2,
                                      running_grads2)]
    ru2up = [(ru2, 0.95*ru2+0.05*(ud**2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p+ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    """
    A variant of SGD that scales the step size by running average of the
    recent step norms.
    :param lr: Theano SharedVariable, initial learning rate.
    :param tparams: Theano SharedVariable, model parameters.
    :param grads: Theano variable, gradients of cost w.r.t. parameters.
    :param x: Theano varaible, model inputs.
    :param mask: Theano variable, sequence mask.
    :param y: Theano variable, targets.
    :param cost: Theano variable, objective function to minimize.
    :return:
    Note: for more information, see Geoff Hinton, Neural networks for
        machine learning, lecture 6a,
        htttp://cs.toronto.edu/~tijmen/csc321/sildes/lecture_slides_lec5.pdf
    """
    zipped_grads = [theano.shared(p.get_value()*numpy_floatX(0.),
                                  name='$s_grad'%k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value()*numpy_floatX(0.),
                                   name='%s_rgrad'%k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value()*numpy_floatX(0.),
                                    name='%s_rgrad'%k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95*rg+0.05*g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95*rg2+0.05*(g**2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(
        [x, mask, y], cost,
        updates=zgup+rgup+rg2up,
        name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value()*numpy_floatX(0.),
                           name='%s_updir'%k)
             for k, p in tparams.items()]
    updir_new =[(ud, 0.9*ud-1e-4*zg/tensor.sqrt(rg2-rg**2+1e-4))
                for ud, zg, rg, rg2 in zip(updir,
                                           zipped_grads,
                                           running_grads,
                                           running_grads2)]
    param_up = [(p, p+udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')
    return f_grad_shared, f_update


def build_model(tparams, options):
    """
    Building LSTM model for sentiment analysis.
    :param tparams: trainable parameters of model.
    :param options: hyper-parameters of model.
    :return:
    """
    # Get rand seed.
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    # Data for model.
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.matrix('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    proj = get_layer(options['encoder'])[1](
        tparams, emb, options, prefix=options['encoder'], mask=mask)
    if options['encoder']=='lstm':
        proj = (proj*mask[:,:,None]).sum(axis=0)
        proj = proj/mask.sum(axis=0)[:,None]
    if options['use_dorpout']:
        proj = dropout_layer(proj, use_noise, trng)

    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U'])+tparams['b'])

    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, maks], pred.argmax(axis=1), name='f_pred')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = -tensor.log(pred[tensor.arange(n_samples), y]+off).mean()

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """
    If you want to to use a trained model, this is useful to compute the
    probabilities of new examples.
    :param f_pred_prob: function for predict probabilities.
    :param prepare_data: function for preparing data.
    :param data: data(imdb).
    :param iterator: the index of data will be calculate probabilities.
    :param verbose: whether print log note.
    :return:
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2).astype(config.floatX))

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    return probs


def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    """
    Calculting the error of prediction.
    :param f_pred: Theano function, computing the prediction.
    :param prepare_data: function for get prepared data.
    :param data: data for prediction.
    :param iterator: the index of data will be calculate prediction error.
    :param verbose: whether print log note.
    :return: the percentage of error prediction.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        preds = f_pred(x, mask)
        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err)/len(data[0])

    return valid_err


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
    # Model options
    model_options = locals().copy()
    print("model options", model_options)

    # Getting data.
    load_data, prepare_data = get_dataset(dataset)
    print("loading data")
    train, valid, test = load_data(
        n_words=n_words, valid_portion=0.05, maxlen=maxlen)

    # Shuffle the data.
    if test_size > 0:
        # The test set is sorted by size, but we want to keep random size
        # example. So we must select a random selection of the examples.
        idx = numpy.arange(len(test[0]))
        numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])

    # Set output dimension.
    ydim = numpy.max(train[1]) + 1
    model_options['ydim'] = ydim

    # Building model.
    print("Building model")
    # This create the initial parameters as numpy ndarrays.
    # Dic name (string)->numpy ndarray
    params = init_params(model_options)

    # Re-load model from file if needed.
    if reload_model:
        load_params('lstm_model.npz', params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable params and
    # tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout.
    (use_noise, x, musk, y, f_pred_prob, f_pred, cost) = \
        build_model(tparams, model_options)

    # If weight decay are used.
    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'**2]).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # Set objective function.
    f_cost = theano.function([x, mask, y], cost, name='f_cost')

    # Set gradient.
    grads = tensor.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    # Set learning rate.
    lr = tensor.scalar(name='lr')

    # Set optimizer.
    f_grad_shared, f_update = optimizer(
        lr, tparams, grads, x, mask, y, cost)

    # Opimization.
    print('Optimization')
    kf_valid = get_mimibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_mimibatches_idx(len(test[0]), valid_batch_size)
    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0])//batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])//batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()

    try:
        for eidx in range(max_epochs):
            n_samples = 0
    except KeyboardInterrupt:
        print("Training interrupted")

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0], batch_size))
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error(f_pred, prepare_data, test, kf_test)

    print('Train', train_err, 'Valid', valid_err, 'Test', test_err)
    if saveto:
        numpy.savez(saveto, train_err=train_err, valid_err=valid_err,
                    test_err=test_err, history_errs=history_errs, **best_p)
    print('The code run for %d epochs, with %f sec/epoch' % (
        (eidx + 1), (end_time - start_time)/(1.*eidx+1)))
    print(('Training took %.fs' % (end_time-start_time)), file=sys.stderr)

    return train_err, valid_err, test_err


if __name__=='__main__':
    # See function train for all possible parameter and there definition.
    train_lstm(max_epochs=100, test_size=500)
