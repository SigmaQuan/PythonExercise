import os as os
import numpy as np


def get_one_file_data(file_name):
    """
    Read babi data set file of each task, and label each
    sentence in the file with
        C: context
        Q: question
        A: answer

    Parameters
    ----------
    :param file_name: text file name, which content training or testing
    sample of each task.
    :return: all the samples for training or testing the task.
    """
    print "==> Loading data from %s" % file_name
    tasks = []
    task = None
    for i, line in enumerate(open(file_name)):
        id = int(line[0:line.find(' ')])
        if id == 1:
            task = {"C": "", "Q": "", "A": ""}

        line = line.strip()
        line = line.replace('.', ' . ')
        line = line[line.find(' ')+1:]
        if line.find('?') == -1:
            task["C"] += line
        else:
            idx = line.find('?')
            tmp = line[idx+1:].split('\t')
            task["Q"] = line[:idx]
            task["A"] = tmp[1].strip()
            tasks.append(task.copy())

    return tasks


def get_raw(folder, train_id, test_id):
    """
    Get babi training and testing data by task id.
    :param train_id: task id for training.
    :param test_id: task id for testing
    :return: the training and testing data.
    """
    babi_map = {
        "1": "qa1_single-supporting-fact",
        "2": "qa2_two-supporting-facts",
        "3": "qa3_three-supporting-facts",
        "4": "qa4_two-arg-relations",
        "5": "qa5_three-arg-relations",
        "6": "qa6_yes-no-questions",
        "7": "qa7_counting",
        "8": "qa8_lists-sets",
        "9": "qa9_simple-negation",
        "10": "qa10_indefinite-knowledge",
        "11": "qa11_basic-coreference",
        "12": "qa12_conjunction",
        "13": "qa13_compound-coreference",
        "14": "qa14_time-reasoning",
        "15": "qa15_basic-deduction",
        "16": "qa16_basic-induction",
        "17": "qa17_positional-reasoning",
        "18": "qa18_size-reasoning",
        "19": "qa19_path-finding",
        "20": "qa20_agents-motivations",
        "joint": "all_shuffled",
        "sh1": "../shuffled/qa1_single-supporting-fact",
        "sh2": "../shuffled/qa2_two-supporting-facts",
        "sh3": "../shuffled/qa3_three-supporting-facts",
        "sh4": "../shuffled/qa4_two-arg-relations",
        "sh5": "../shuffled/qa5_three-arg-relations",
        "sh6": "../shuffled/qa6_yes-no-questions",
        "sh7": "../shuffled/qa7_counting",
        "sh8": "../shuffled/qa8_lists-sets",
        "sh9": "../shuffled/qa9_simple-negation",
        "sh10": "../shuffled/qa10_indefinite-knowledge",
        "sh11": "../shuffled/qa11_basic-coreference",
        "sh12": "../shuffled/qa12_conjunction",
        "sh13": "../shuffled/qa13_compound-coreference",
        "sh14": "../shuffled/qa14_time-reasoning",
        "sh15": "../shuffled/qa15_basic-deduction",
        "sh16": "../shuffled/qa16_basic-induction",
        "sh17": "../shuffled/qa17_positional-reasoning",
        "sh18": "../shuffled/qa18_size-reasoning",
        "sh19": "../shuffled/qa19_path-finding",
        "sh20": "../shuffled/qa20_agents-motivations",
    }

    if (test_id == ""):
        test_id = train_id
    babi_train_file_name = babi_map[train_id]
    babi_test_file_name = babi_map[test_id]
    # babi_train_raw = get_one_file_data(
    #     os.path.join(os.path.dirname(os.path.realpath(__file__)),
    #                  'data/en/%s_train.txt' % babi_train_file_name))
    # babi_test_raw = get_one_file_data(
    #     os.path.join(os.path.dirname(os.path.realpath(__file__)),
    #                  'data/en/%s_test.txt' % babi_test_file_name))
    babi_train_raw = get_one_file_data(
        os.path.join(folder, 'en/%s_train.txt' % babi_train_file_name))
    babi_test_raw = get_one_file_data(
        os.path.join(folder, 'en/%s_test.txt' % babi_test_file_name))
    return babi_train_raw, babi_test_raw
