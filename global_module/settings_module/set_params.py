class ParamsClass:
    def __init__(self, mode='TR'):
        """
        :param mode: 'TR' for train, 'TE' for test, 'VA' for valid
        """
        self.mode = mode
        self.init_scale = 0.1
        self.learning_rate = 0.001
        self.max_grad_norm = 10
        self.max_epoch = 10
        self.max_max_epoch = 15
        self.rnn_cell = 'lstm'  # or 'gru' OR 'lstm'
        self.bidirectional = True

        if mode == 'TR':
            self.keep_prob = 0.5
        else:
            self.keep_prob = 1.0

        self.lr_decay = 0.99

        self.enable_shuffle = True
        self.enable_checkpoint = False
        self.all_lowercase = True
        self.log = False
        self.log_step = 9

        if mode == 'TE':
            self.enable_shuffle = False

        self.NUM_LAYER = 1
        self.REG_CONSTANT = 0.1
        self.MAX_SEQ_LEN = 150
        self.EMB_DIM = 200
        self.RNN_HIDDEN_DIM = 256
        self.ATTENTION_DIM = 256

        self.batch_size = 1000
        self.VOCAB_SIZE = 30
        self.IS_WORD_TRAINABLE = False

        self.use_unknown_word = False
        self.use_random_initializer = False

        self.use_attention = True

        self.indices = None
        self.num_instances = None
        self.num_classes = None
        self.sampling_threshold = 10

        self.ASPECT_NUM = 14
        self.optimizer = 'adam'
