import numpy as np
import sklearn.metrics as metrics

import babi
# import argparse
import json

import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class TrainOneTask:
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


    def train(self, acc_delta):
        fig = plt.figure()
        plt.axis([0, 10, 0, 1])
        plt.ion()

        if self.args.mode == 'train':
            print "==> training"
            skipped = 0
            # train_error = []
            # test_error = []
            train_acc = []
            test_acc = []
            epochs = []
            need_break = 0
            for epoch in range(self.args.epochs):
                start_time = time.time()

                if self.args.shuffle:
                    self.dam.shuffle_train_set()

                training_acc, training_loss, skipped = self.do_epoch('train', epoch, skipped)
                testing_acc, test_loss, skipped = self.do_epoch('test', epoch, skipped)
                # training_acc = np.random.randn()
                # testing_acc = np.random.randn()

                # state_name = 'states/%s.epoch%d.test%.5f.state' % (self.network_name, epoch, test_loss)
                #
                # if (epoch % self.args.save_every == 0):
                #     print "==> saving ... %s" % state_name
                #     self.dmn.save_params(state_name, epoch)

                print "epoch %d took %.3fs" % (epoch, float(time.time()) - start_time)

                # print "show weight matrix"
                # self.dam.show_weight()
                self.dam.print_input_module()

                # train_error.append(training_loss)
                # test_error.append(test_loss)
                train_acc.append(training_acc)
                test_acc.append(testing_acc)
                epochs.append(epoch)
                plt.gca().cla()
                plt.plot(epochs, train_acc, 'r.-', label="Train")
                plt.plot(epochs, test_acc, 'g.-', label="Test")
                plt.xlabel('epoch')
                plt.ylabel('accuracy')
                plt.legend(loc=4)
                self.network_name = 'task_%s.epoch_%03d.train_%06.3f.test_%06.3f' % (
                    self.args.babi_id, epoch, np.max(train_acc), np.max(test_acc))
                plt.title(self.network_name)
                plt.draw()
                plt.grid(True)
                plt.pause(0.05)

                if (training_acc > acc_delta and testing_acc > acc_delta):
                    need_break = need_break + 1

                if need_break > 10:
                    print "==> need_break > 10"
                    break

                if (training_acc > 95.0 and testing_acc < 47.5 and
                            test_acc[len(test_acc) - 1] < test_acc[len(test_acc) - 2]):
                    print "==> training_acc > %.5f and testing_acc < %.5f and test_acc[n] %.5f < test_acc[n-1]%.5f" % (
                        training_acc, testing_acc, test_acc[len(test_acc) - 1], test_acc[len(test_acc) - 2])
                    break

                if (len(test_acc) > 40 and
                            np.std(train_acc[len(train_acc) - 30: len(train_acc) - 1]) < 0.5 and
                            np.std(test_acc[len(test_acc) - 30: len(test_acc) - 1]) < 0.5):
                    print "==> parallel"
                    break

            print ('babi task id: %2d' % self.args.babi_id)
            print ('epoch: %2d' % len(epochs))
            print ('training acc.: %f' % np.max(train_acc))
            print ('testing acc.: %f' % np.max(test_acc))

            state_name = '%s/%s.state'%(self.folder_name, self.network_name)
            print "==> saving ... %s" % state_name
            self.dam.save_params(state_name, epoch)

            imageName = '%s/task_%s.pdf' % (self.folder_name, self.args.babi_id)
            pp = PdfPages(imageName)
            plt.savefig(pp, format='pdf')
            pp.close()
            plt.close()

        elif self.args.mode == 'test':
            file = open('last_tested_model.json', 'w+')
            data = dict(self.args._get_kwargs())
            data["id"] = self.network_name
            data["name"] = self.network_name
            data["description"] = ""
            data["vocab"] = self.dam.vocab.keys()
            json.dump(data, file, indent=2)
            self.do_epoch('test', 0)

        else:
            raise Exception("unknown mode")
