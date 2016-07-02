import os as os
import numpy as np


def load_glove(dim):
    word2vec = {}
    print "==> loading glove"
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           "data/glove/glove.6B." + str(dim) + "d.txt")) as f:
        for line in f:
            l = line.split()
            word2vec[l[0]] = map(float, l[1:])

    print "==> glove is loaded"
    return word2vec


def create_vector(word, word2vec, word_vector_size, silent=False):
    # if the word is missing from Glove, create some fake vector and store in glove!
    vector = np.random.uniform(0.0, 1.0, (word_vector_size,))
    word2vec[word] = vector
    if (not silent):
        print "utils.py::create_vector => %s is missing" % word
    return vector


def process_word(word, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=False):
    if not word in word2vec:
        create_vector(word, word2vec, word_vector_size, silent)
    if not word in vocab:
        next_index = len(vocab)
        vocab[word] = next_index
        ivocab[next_index] = word

    if to_return == "word2vec":
        return word2vec[word]
    elif to_return == "index":
        return vocab[word]
    elif to_return == "onehot":
        raise Exception("to_return = 'onehot' is not implemented yet")