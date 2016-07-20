import os as os
import numpy as np


def parser_one_file(file_name):
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
    removed_string_1 = "www.freebase.com/m/"
    removed_string_2 = "www.freebase.com/"
    removed_character_1 = '\n'
    removed_character_2 = '?'
    removed_character_3 = '.'
    removed_character_4 = ''
    samples = []
    sample = None
    for i, line in enumerate(open(file_name)):
        line = line.replace(removed_string_1, removed_character_4)
        line = line.replace(removed_string_2, removed_character_4)

        line = line.rstrip(removed_character_1)
        line = line.rstrip(removed_character_2)
        line = line.rstrip(removed_character_3)

        line = line.lower()

        tokens = line.split('\t')

        # print tokens
        if len(tokens) != 4:
            print line
            continue

        sample = {"subject": "", "relationship": "", "object": "", "question": ""}
        sample["subject"] = tokens[0].strip()
        # sample["relationship"] = tokens[1].split('/')
        sample["relationship"] = tokens[1].strip()
        sample["object"] = tokens[2].strip()
        sample["question"] = tokens[3].split(' ')

        samples.append(sample.copy())

    return samples


def get_raw(folder):
    """
    Get babi training and testing data by task id.
    :param folder: the folder, which contain three data files.
    :return: the training and testing data.
    """
    # the default file name
    simple_qa_map = {
        "train": "annotated_fb_data_train.txt",
        "valid": "annotated_fb_data_valid.txt",
        "test": "annotated_fb_data_test.txt"
    }

    train_file_name = folder + simple_qa_map["train"]
    valid_file_name = folder + simple_qa_map["valid"]
    test_file_name = folder + simple_qa_map["test"]
    train_raw = parser_one_file(train_file_name)
    valid_raw = parser_one_file(valid_file_name)
    test_raw = parser_one_file(test_file_name)

    total_sample = 108442
    train_sample = 75910
    valid_sample = 10845
    test_sample = total_sample - train_sample - valid_sample

    if ((len(train_raw) == train_sample) and
            (len(valid_raw) == valid_sample) and
            (len(test_raw) == test_sample)):
        return train_raw, valid_raw, test_raw
    else:
        print "Read file error!"
        return None, None, None


def get_one_raw_dictionaries(raw_data):
    # entity name
    entity_dict = {}
    entity_list = []

    # relationship
    relationship_dict = {}
    relationship_list = []
    relationship_word_dict = {}
    relationship_word_list = []

    # question
    question_word_dict = {}
    question_word_list = []

    # deal with each sample
    for sample in raw_data:
        # print sample
        entity_1 = sample["subject"]
        if entity_1 not in entity_dict:
            entity_dict[entity_1] = 1
        else:
            entity_dict[entity_1] += 1

        relationship = sample["relationship"]
        if relationship not in relationship_dict:
            relationship_dict[relationship] = 1
        else:
            relationship_dict[relationship] += 1

        # for word in sample["relationship"]:
        #     if word not in relationship_dict:
        #         relationship_dict[word] = 1
        #     else:
        #         relationship_dict[word] += 1

        entity_2 = sample["object"]
        if entity_2 not in entity_dict:
            entity_dict[entity_2] = 1
        else:
            entity_dict[entity_2] += 1

        for word in sample["question"]:
            if word not in question_word_dict:
                question_word_dict[word] = 1
            else:
                question_word_dict[word] += 1

    print "entity_dict size = %d" % (len(entity_dict))
    entity_dict = sorted(entity_dict.items(), key=lambda d: d[1], reverse=True)
    # print entity_dict

    print "relationship_dict size = %d" % (len(relationship_dict))
    # relationship_dict = sorted(relationship_dict.items(), key=lambda d: d[1], reverse=True)
    print relationship_dict
    print sorted(relationship_dict.items(), key=lambda d: d[1], reverse=True)
    print sorted(relationship_dict.items(), key=lambda d: d[0], reverse=True)

    print "question_word_dict size = %d" % (len(question_word_dict))
    question_word_dict = sorted(question_word_dict.items(), key=lambda d: d[1], reverse=True)
    # print question_word_dict

def get_dictionaries(train_raw, valid_raw, test_raw):
    # entity name
    entity_dict = {}
    entity_list = []

    # relationship
    relationship_dict = {}
    relationship_list = []
    relationship_word_dict = {}
    relationship_word_list = []

    # question
    question_word_dict = {}
    question_word_list = []

    for sample in train_raw:
        # print sample
        entity_1 = sample["subject"]
        if entity_1 not in entity_dict:
            entity_dict[entity_1] = 1
        else:
            entity_dict[entity_1] += 1

        relationship = sample["relationship"]
        if relationship not in relationship_dict:
            relationship_dict[relationship] = 1
        else:
            relationship_dict[relationship] += 1

        # for word in sample["relationship"]:
        #     if word not in relationship_dict:
        #         relationship_dict[word] = 1
        #     else:
        #         relationship_dict[word] += 1

        entity_2 = sample["object"]
        if entity_2 not in entity_dict:
            entity_dict[entity_2] = 1
        else:
            entity_dict[entity_2] += 1

        for word in sample["question"]:
            if word not in question_word_dict:
                question_word_dict[word] = 1
            else:
                question_word_dict[word] += 1

    print "entity_dict size = %d" % (len(entity_dict))
    entity_dict = sorted(entity_dict.items(), key=lambda d: d[1], reverse=True)
    # print entity_dict

    print "relationship_dict size = %d" % (len(relationship_dict))
    # relationship_dict = sorted(relationship_dict.items(), key=lambda d: d[1], reverse=True)
    print relationship_dict
    print sorted(relationship_dict.items(), key=lambda d: d[1], reverse=True)
    print sorted(relationship_dict.items(), key=lambda d: d[0], reverse=True)

    print "question_word_dict size = %d" % (len(question_word_dict))
    question_word_dict = sorted(question_word_dict.items(), key=lambda d: d[1], reverse=True)
    # print question_word_dict
