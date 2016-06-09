import theano
import theano.tensor as T
import lasagne
import numpy as np
import os


# Get the all files & directories in the specified directory (path).
def get_recursive_file_list(path):
    current_files = os.listdir(path)
    all_files = []
    for file_name in current_files:
        full_file_name = os.path.join(path, file_name)
        all_files.append(full_file_name)

        if os.path.isdir(full_file_name):
            next_level_files = get_recursive_file_list(full_file_name)
            all_files.extend(next_level_files)

    return all_files


def softmax(x):
    e_x = T.exp(x - x.max(axis=0, keepdims=True))
    out = e_x / e_x.sum(axis=0, keepdims=True)
    return out


def l2_reg(params):
    return T.sum([T.sum(x ** 2) for x in params])


def constant_param(value=0.0, shape=(0,)):
    return theano.shared(lasagne.init.Constant(value).sample(shape), borrow=True)


def normal_param(std=0.1, mean=0.0, shape=(0,)):
    return theano.shared(lasagne.init.Normal(std, mean).sample(shape), borrow=True)


def uniform_param(std=0.01, mean=0.0, shape=(0,)):
    return theano.shared(lasagne.init.Uniform(std, mean).sample(shape), borrow=True)


def get_norm(x):
    x = np.array(x)
    return np.sum(x * x)
