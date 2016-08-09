import util
import numpy as np
import os
import sys
import time

import word
from hyperparameter import HyperParameterSpace
from babi_task_training import TrainOneTask

from hyperparameter import HyperParameterSpace
from babi_task_testing import TestOneTask

from util import get_recursive_file_list


def test_once(babi_id, load_state, word2vec):
    print "==> get one argument for each task"
    hyper_parameter = HyperParameterSpace()
    args = hyper_parameter.produce_one_arg()
    args.babi_id = babi_id
    # args.load_state = "task_1.epoch_056.train_96.200.test_51.800.state"
    # args.load_state = "task_1.epoch_080.train_100.000.test_99.900.state"
    args.load_state = load_state
    args.mode = "test"

    # print "==> testing each task.."
    # folder_name = time.strftime('experiment/test-%Y-%m-%d-%H-%M-%S')
    folder_name = time.strftime('experiment/visualization-%Y-%m-%d-%H-%M-%S')
    folder_name = (folder_name + ('task-%s' % babi_id))
    print "setting folder: %s" % folder_name
    os.makedirs(folder_name)
    i = 1
    # print "Testing task %d..." % i
    print "Testing task %s..." % babi_id
    start_time = time.time()
    sys_stdout = sys.stdout
    # log_file = '%s/task_%02d.log' % (folder_name, i+1)
    log_file = '%s/task_%s.log' % (folder_name, babi_id)
    sys.stdout = open(log_file, 'a')
    # print "Training task %d..." % i
    print args
    test_one_task = TestOneTask(args, word2vec)
    test_one_task.set_folder_name(folder_name)
    # test_one_task.test()
    test_one_task.visualization()
    # print "task %d took %.3fs" % (i, float(time.time()) - start_time)
    print "task %s took %.3fs" % (babi_id, float(time.time()) - start_time)
    sys.stdout.close()
    sys.stdout = sys_stdout
    # print "task %d took %.3fs" % (i, float(time.time()) - start_time)
    print "task %s took %.3fs" % (babi_id, float(time.time()) - start_time)


    print "==> end testing."


def test():
    babi_id = "1"
    # load_state = "task_1.epoch_056.train_96.200.test_51.800.state"
    # load_state = "task_1.epoch_080.train_100.000.test_99.900.state"
    load_state = "state/task_1.state"
    test_once(babi_id, load_state)


def visualization_weight():
    state_dir = "Best.Model.Visualization"
    load_state = get_recursive_file_list(state_dir)
    print load_state
    # ['Best.Model.Visualization/task_16.state',
    #  'Best.Model.Visualization/task_18_test_37.state',
    #  'Best.Model.Visualization/task_06.state',
    #  'Best.Model.Visualization/task_20_test_05.state',
    #  'Best.Model.Visualization/task_07.state',
    #  'Best.Model.Visualization/task_08.state',
    #  'Best.Model.Visualization/task_11_test_06.state',
    #  'Best.Model.Visualization/task_20_test_37.state',
    #  'Best.Model.Visualization/task_05_test_20.state',
    #  'Best.Model.Visualization/task_01.state',
    #  'Best.Model.Visualization/task_09.state',
    #  'Best.Model.Visualization/task_20_test_38.state',
    #  'Best.Model.Visualization/task_18_test_12.state',
    #  'Best.Model.Visualization/task_10.state',
    #  'Best.Model.Visualization/task_11_test_37.state',
    #  'Best.Model.Visualization/task_02.state',
    #  'Best.Model.Visualization/task_15_test_11.state',
    #  'Best.Model.Visualization/task_14_test_14.state',
    #  'Best.Model.Visualization/task_17.state',
    #  'Best.Model.Visualization/task_05_test_37.state',
    #  'Best.Model.Visualization/task_12.state',
    #  'Best.Model.Visualization/task_14_test_36.state',
    #  'Best.Model.Visualization/task_13.state',
    #  'Best.Model.Visualization/task_04.state',
    #  'Best.Model.Visualization/task_19.state',
    #  'Best.Model.Visualization/task_03.state']

    # babi_id = ["1", "2", "3", "4", "5", "5",
    #            "6", "7", "8", "9", "10",
    #            "11", "11", "12", "13", "14", "14", "15",
    #            "16", "17", "18", "18", "19", "20", "20", "20"]

    babi_id = ["16", "18", "6", "20", "7", "8",
               "11", "20", "5", "1", "9",
               "20", "18", "10", "11", "2", "15", "14",
               "17", "5", "12", "14", "13", "4", "19", "3"]

    print len(load_state)
    print len(babi_id)

    assert len(load_state) == len(babi_id)

    # load word vector: glove
    word2vec = word.load_glove(50)

    for i in range(len(load_state)):
        test_once(babi_id[i], load_state[i], word2vec)


    print("Finished!")

if __name__ == "__main__":
    # test()
    visualization_weight()

