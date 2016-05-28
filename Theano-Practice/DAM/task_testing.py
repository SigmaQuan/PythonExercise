import numpy as np
import sklearn.metrics as metrics

import babi
# import argparse
import json

import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class TestOneTask:
    def __init__(self, arg, word_vector):
        self.args = arg
        self.word2vec = word_vector
        self.network_name = self.set_network_name()
        # get data of the task
        self.babi_train_raw, self.babi_test_raw = babi.get_raw(
            self.args.babi_id, self.args.babi_test_id)
        self.args_dict = dict(self.args._get_kwargs())
        print self.args_dict
        self.args_dict['babi_train_raw'] = self.babi_train_raw
        self.args_dict['babi_test_raw'] = self.babi_test_raw
        self.args_dict['word2vec'] = self.word2vec
        self.dam = self.set_dam()
        if self.args.load_state != "":
            self.dam.load_state(self.args.load_state)

        self.folder_name = time.strftime('%Y-%m-%d %H:%M:%S')

    def set_folder_name(self, folder_name):
        self.folder_name = folder_name


    # set network name
    def set_network_name(self):
        return self.args.prefix + '%s.mh%d.n%d.bs%d%s%s%s.babi%s' % (
            self.args.network,
            self.args.memory_hops,
            self.args.dim,
            self.args.batch_size,
            ".na" if self.args.normalize_attention else "",
            ".bn" if self.args.batch_norm else "",
            (".d" + str(self.args.dropout)) if self.args.dropout > 0 else "",
            self.args.babi_id)

    # init class
    def set_dam(self):
        if self.args.network == 'dmn_batch':
            import dam
            return dam.DAM(**self.args_dict)
        # elif self.args.network == 'dmn_basic':
        #     import dmn_basic
        #     if (self.args.batch_size != 1):
        #         print "==> no minibatch training, argument batch_size is useless"
        #         self.args.batch_size = 1
        #     return dmn_basic.DMN_basic(**self.args_dict)
        # elif self.args.network == 'dmn_smooth':
        #     import dmn_smooth
        #     if (self.args.batch_size != 1):
        #         print "==> no minibatch training, argument batch_size is useless"
        #         self.args.batch_size = 1
        #     return dmn_smooth.DMN_smooth(**self.args_dict)
        # elif self.args.network == 'dmn_qa':
        #     import dmn_qa_draft
        #     if (self.args.batch_size != 1):
        #         print "==> no minibatch training, argument batch_size is useless"
        #         self.args.batch_size = 1
        #     return dmn_qa_draft.DMN_qa(**self.args_dict)
        else:
            raise Exception("No such network known: " + self.args.network)


    def do_epoch(self, mode, epoch, skipped=0):
        y_true = []
        y_pred = []
        avg_loss = 0.0
        prev_time = time.time()

        batches_per_epoch = self.dam.get_batches_per_epoch(mode)

        for i in range(0, batches_per_epoch):
            step_data = self.dam.step(i, mode)
            prediction = step_data["prediction"]
            answers = step_data["answers"]
            current_loss = step_data["current_loss"]
            current_skip = (step_data["skipped"] if "skipped" in step_data else 0)
            log = step_data["log"]

            skipped += current_skip

            if current_skip == 0:
                avg_loss += current_loss

                for x in answers:
                    y_true.append(x)

                for x in prediction.argmax(axis=1):
                    y_pred.append(x)

                # TODO: save the state sometimes
                if (i % self.args.log_every == 0):
                    cur_time = time.time()
                    print ("  %sing: %d.%d / %d \t loss: %.3f \t avg_loss: %.3f \t skipped: %d \t %s \t time: %.2fs" %
                           (mode, epoch, i * self.args.batch_size, batches_per_epoch * self.args.batch_size,
                            current_loss, avg_loss / (i + 1), skipped, log, cur_time - prev_time))
                    prev_time = cur_time

            if np.isnan(current_loss):
                print "==> current loss IS NaN. This should never happen :) "
                exit()

        avg_loss /= batches_per_epoch
        print "\n  %s loss = %.5f" % (mode, avg_loss)
        print "confusion matrix:"
        print metrics.confusion_matrix(y_true, y_pred)

        accuracy = sum([1 if t == p else 0 for t, p in zip(y_true, y_pred)])
        accuracy = (accuracy * 100.0 / batches_per_epoch / self.args.batch_size)
        print "accuracy: %.2f percent" % accuracy

        return accuracy, avg_loss, skipped

    def test(self):
        # self.dam.show_input_module()
        # self.dam.show_memory_module()
        self.dam.show_weight()
        file = open('last_tested_model.json', 'w+')
        data = dict(self.args._get_kwargs())
        data["id"] = self.network_name
        data["name"] = self.network_name
        data["description"] = ""
        data["vocab"] = self.dam.vocab.keys()
        json.dump(data, file, indent=2)
        self.do_epoch('test', 1)
