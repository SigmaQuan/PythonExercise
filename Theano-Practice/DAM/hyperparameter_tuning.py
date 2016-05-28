import os
import sys
import time

import word
from hyperparameter import HyperParameterSpace
from task_training import TrainOneTask

print "==> get one argument for each task"
hyper_parameter = HyperParameterSpace()
# print hyper_parameter.answer_module
# hyper_parameter.cout()
# args = hyper_parameter.produce_one_arg()
# print args
arg_list = hyper_parameter.produce_arg_list_by_tasks()

# load word vector: glove
# word2vec = utils.load_glove(arg_list[0].word_vector_size)
word2vec = word.load_glove(arg_list[0].word_vector_size)
# babi_train_raw, babi_test_raw = babi.get_raw("1", "1")

print "==> training each task.."

acc_delta = [99.9, 99.9, 99.9, 99.9, 99.5,
             99.9, 97.5, 97.0, 99.9, 98.0,
             99.9, 99.9, 99.9, 99.9, 99.9,
             99.9, 66.0, 95.9, 40.0, 99.9]

folder_name = time.strftime('experiment/%Y-%m-%d-%H-%M-%S')
os.makedirs(folder_name)
# print acc_delta
for i in range(0, 20, 1):
    print "Training task %d..." % i
    start_time = time.time()
    sys_stdout = sys.stdout
    log_file = '%s/task_%02d.log' % (folder_name, i+1)
    sys.stdout = open(log_file, 'a')
    print "Training task %d..." % i
    print arg_list[i]
    train_one_task = TrainOneTask(arg_list[i], word2vec)
    train_one_task.set_folder_name(folder_name)
    train_one_task.train(acc_delta[i])
    print "task %d took %.3fs" % (i, float(time.time()) - start_time)
    sys.stdout.close()
    sys.stdout = sys_stdout
    print "task %d took %.3fs" % (i, float(time.time()) - start_time)

print "==> end training."

