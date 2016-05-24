import util
import numpy as np
import os
import sys
import time

import word
from hyperparameter import HyperParameterSpace
from task_training import TrainOneTask

print "==> get one argument for each task"
from hyperparameter import HyperParameterSpace
from task_testing import TestOneTask

# w1 = util.uniform_param(std=100, shape=(100, 200))
# print(w1)
# w = np.random.random((200, 300))
# print(w1.get_value())
# print(w)

hyper_parameter = HyperParameterSpace()
args = hyper_parameter.produce_one_arg()
args.babi_id = "1"
args.load_state = "/Documents/Code/PythonExercise/Theano-Practice/DAM/experiment/task_1.epoch_025.train_100.000.test_100.000.state"


# load word vector: glove
# word2vec = utils.load_glove(arg_list[0].word_vector_size)
word2vec = word.load_glove(args.word_vector_size)

print "==> testing each task.."

folder_name = time.strftime('experiment/test-%Y-%m-%d %H:%M:%S')
os.makedirs(folder_name)
# print acc_delta
for i in range(0, 1, 1):
    print "Testing task %d..." % i
    start_time = time.time()
    sys_stdout = sys.stdout
    log_file = '%s/task_%02d.log' % (folder_name, i+1)
    sys.stdout = open(log_file, 'a')
    print "Training task %d..." % i
    print args
    test_one_task = TestOneTask(args, word2vec)
    test_one_task.set_folder_name(folder_name)
    test_one_task.test()
    print "task %d took %.3fs" % (i, float(time.time()) - start_time)
    sys.stdout.close()
    sys.stdout = sys_stdout
    print "task %d took %.3fs" % (i, float(time.time()) - start_time)

print "==> end testing."