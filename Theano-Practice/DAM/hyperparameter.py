import argparse


class HyperParameterSpace:
    def __init__(self):
        self.network = ["dmn_batch", "dmn_basic", "dmn_smooth", "dmn_qa"]
        self.word_vector_size = [50, 100, 200, 300]
        self.dim = [40, 80, 160, 320]
        self.epochs = [100, 150, 200, 300, 500]
        self.load_state = ""
        self.answer_module = ["feedforward", "recurrent"]
        self.mode = ["train", "test"]
        self.input_mask_mode = ["sentence", "word"]
        self.memory_hops = [3, 5, 10, 15, 30, 60]
        self.batch_size = [5, 10, 15, 20]
        self.babi_id = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                        "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
        self.l2 = [0, 0.001, 0.1]
        self.normalize_attention = [True, False]
        self.log_every = [0, 1]
        self.save_every = [0, 1]
        self.prefix = ""
        self.shuffle = True
        self.babi_test_id = ""
        self.dropout = [0, 0.1, 0.3, 0.5]
        self.batch_norm = False

    def cout(self):
        print "network name"
        print self.network
        print "word vector size"
        print self.word_vector_size
        print "dim: number of hidden units in input module GRU"
        print self.dim
        print "epochs"
        print self.epochs
        print "load state"
        print self.load_state
        print "answer module"
        print self.answer_module
        print "mode"
        print self.mode
        print "input mask mode"
        print self.input_mask_mode
        print "memory hops"
        print self.memory_hops
        print "batch size"
        print self.batch_size
        print "babi task id"
        print self.babi_id
        print "l2 normalization"
        print self.l2
        print "normalize attention"
        print self.normalize_attention
        print "log every"
        print self.log_every
        print "save every"
        print self.save_every
        print "prefix"
        print self.prefix
        print "shuffle"
        print self.shuffle
        print "babi test id"
        print self.babi_test_id
        print "dropout rate"
        print self.dropout
        print "batch normalization"
        print self.batch_norm

    def produce_one_arg(self):
        # parser = argparse.ArgumentParser()
        # parser.add_argument('--network', type=str, default=self.network[0],
        #                     help='"dmn_batch", "dmn_basic", "dmn_smooth", "dmn_qa"')
        # parser.add_argument('--word_vector_size', type=int, default=self.word_vector_size[0],
        #                     help='embeding size (50, 100, 200, 300 only)')
        # parser.add_argument('--dim', type=int, default=self.dim[0],
        #                     help='number of hidden units in input module GRU')
        # parser.add_argument('--epochs', type=int, default=self.epochs[3],
        #                     help='number of epochs')
        # parser.add_argument('--load_state', type=str, default="", help='state file path')
        # parser.add_argument('--answer_module', type=str, default=self.answer_module[0],
        #                     help='answer module type: feedforward or recurrent')
        # parser.add_argument('--mode', type=str, default=self.mode[0],
        #                     help='mode: train or test. Test mode required load_state')
        # parser.add_argument('--input_mask_mode', type=str, default=self.input_mask_mode[0],
        #                     help='input_mask_mode: word or sentence')
        # parser.add_argument('--memory_hops', type=int, default=self.memory_hops[1],
        #                     help='memory GRU steps')
        # parser.add_argument('--batch_size', type=int, default=self.batch_size[1],
        #                     help='batch size')
        # parser.add_argument('--babi_id', type=str, default=self.babi_id[0],
        #                     help='babi task ID')
        # parser.add_argument('--l2', type=float, default=self.l2[0],
        #                     help='L2 regularization')
        # parser.add_argument('--normalize_attention', type=bool, default=self.normalize_attention[1],
        #                     help='flag for enabling softmax on attention vector')
        # parser.add_argument('--log_every', type=int, default=self.log_every[1],
        #                     help='print information every x iteration')
        # parser.add_argument('--save_every', type=int, default=self.save_every[1],
        #                     help='save state every x epoch')
        # parser.add_argument('--prefix', type=str, default="",
        #                     help='optional prefix of network name')
        # parser.add_argument('--no-shuffle', dest='shuffle', action='store_false')
        # parser.add_argument('--babi_test_id', type=str, default="",
        #                     help='babi_id of test set (leave empty to use --babi_id)')
        # parser.add_argument('--dropout', type=float, default=self.dropout[2],
        #                     help='dropout rate (between 0 and 1)')
        # parser.add_argument('--batch_norm', type=bool, default=self.batch_norm,
        #                     help='batch normalization')
        # parser.set_defaults(shuffle=self.shuffle)
        parser = argparse.ArgumentParser()
        parser.add_argument('--network', type=str, default="dmn_batch",
                            help='network type: dmn_basic, dmn_smooth, or dmn_batch')
        parser.add_argument('--word_vector_size', type=int, default=50,
                            help='embeding size (50, 100, 200, 300 only)')
        parser.add_argument('--dim', type=int, default=40,
                            help='number of hidden units in input module GRU')
        parser.add_argument('--epochs', type=int, default=500,
                            help='number of epochs')
        parser.add_argument('--load_state', type=str, default="",
                            help='state file path')
        parser.add_argument('--answer_module', type=str, default="feedforward",
                            help='answer module type: feedforward or recurrent')
        parser.add_argument('--mode', type=str, default="train",
                            help='mode: train or test. Test mode required load_state')
        parser.add_argument('--input_mask_mode', type=str, default="sentence",
                            help='input_mask_mode: word or sentence')
        parser.add_argument('--memory_hops', type=int, default=5,
                            help='memory GRU steps')
        parser.add_argument('--batch_size', type=int, default=10,
                            help='no commment')
        parser.add_argument('--babi_id', type=str, default="1",
                            help='babi task ID')
        parser.add_argument('--l2', type=float, default=0,
                            help='L2 regularization')
        parser.add_argument('--normalize_attention', type=bool, default=False,
                            help='flag for enabling softmax on attention vector')
        parser.add_argument('--log_every', type=int, default=1,
                            help='print information every x iteration')
        parser.add_argument('--save_every', type=int, default=1,
                            help='save state every x epoch')
        parser.add_argument('--prefix', type=str, default="",
                            help='optional prefix of network name')
        parser.add_argument('--no-shuffle', dest='shuffle', action='store_false')
        parser.add_argument('--babi_test_id', type=str, default="",
                            help='babi_id of test set (leave empty to use --babi_id)')
        parser.add_argument('--dropout', type=float, default=0.0,
                            help='dropout rate (between 0 and 1)')
        parser.add_argument('--batch_norm', type=bool, default=False,
                            help='batch normalization')
        parser.set_defaults(shuffle=True)
        return parser.parse_args()

    def produce_arg_list_by_tasks(self):
        arg_list = []
        for task in self.babi_id:
            arg = self.produce_one_arg()
            arg.babi_id = task
            arg_list.append(arg)
        return arg_list



