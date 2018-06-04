import numpy as np
import random
from global_module.implementation_module import DataReader, Model
from global_module.settings_module import Directory, Dictionary, Params
import tensorflow as tf
from sklearn.cluster import KMeans
import sys, time, os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class Train:
    def run_epoch(self, session, min_cost, model_obj, dict_obj, epoch_num, reader, lr=0.01, verbose=False):
        combined_loss = -1.0
        epoch_combined_loss = 0.0
        total_instances = 0.0
        step = 0
        print('\nrun epoch')

        params = model_obj.params
        dir_obj = model_obj.dir
        data_filename = dir_obj.data_filename

        output = ''
        data_lines = open(data_filename).readlines()

        for step, (pos_seq_arr, pos_seq_len, neg_seq_arr, neg_seq_len) \
                in enumerate(reader.data_iterator(params, data_filename, model_obj.params.indices, dict_obj)):
            feed_dict = {}

            if len(pos_seq_arr) != params.batch_size:
                print 'WRONG'

            if len(pos_seq_len) != params.batch_size:
                print 'Wrong'

            if len(neg_seq_arr) != params.batch_size:
                print 'Wrong'

            if len(neg_seq_len) != params.batch_size:
                print 'Wrong'

            feed_dict[model_obj.input_utterance] = pos_seq_arr
            feed_dict[model_obj.inp_seq_len] = pos_seq_len
            feed_dict[model_obj.negative_utterance] = neg_seq_arr
            feed_dict[model_obj.neg_seq_len] = neg_seq_len
            feed_dict[model_obj.lr] = lr
            feed_dict[model_obj.keep_prob] = model_obj.params.keep_prob

            try:
                total_loss, regularized_loss, pos_loss, neg_loss, encoded_rep, decoded_rep, neg_avg_rep, pos_avg_rep, aspect_emb, aspect_dot, _ = session.run([model_obj.total_loss,
                                                                                                                                                               model_obj.regularized_loss,
                                                                                                                                                               model_obj.pos_loss,
                                                                                                                                                               model_obj.neg_loss,
                                                                                                                                                               model_obj.encoded_rep,
                                                                                                                                                               model_obj.decoded_rep,
                                                                                                                                                               model_obj.neg_avg_sentence_rep,
                                                                                                                                                               model_obj.pos_avg_sentence_rep,
                                                                                                                                                               model_obj.aspect_emb_matrix,
                                                                                                                                                               model_obj.aspect_dot,
                                                                                                                                                               model_obj.train_op],
                                                                                                                                                              feed_dict=feed_dict)
            except:
                print 'ERROR'

            total_instances += params.batch_size
            epoch_combined_loss += total_loss

            if step % 100 == 0:
                print 'Step: %d, Batch loss: %.4f, Regularized loss: %.7f' % (step, total_loss, regularized_loss)
                if (str(total_loss) == 'nan' or str(regularized_loss) == 'nan'):
                    print 'NaN detected!'

        mean_loss = epoch_combined_loss / step
        print 'Epoch Num: %d, CE loss: %.4f' % (epoch_num, mean_loss)

        if (params.mode == 'VA'):
            model_saver = tf.train.Saver()
            print('**** Current minimum on valid set: %.4f ****' % (min_cost))

            if (mean_loss < min_cost):
                min_cost = mean_loss
                model_saver.save(session, save_path=dir_obj.model_path + dir_obj.model_name, latest_filename=dir_obj.latest_checkpoint)
                print('==== Model saved! ====')

            valid_output = open(dir_obj.output_path + '/valid_op.txt', 'w')
            valid_output.write(output)
            valid_output.close()

        return mean_loss, min_cost, min_cost

    def get_aspect_words(self, model_obj, vocab_map):

        with tf.Session() as session:
            ckpt = tf.train.get_checkpoint_state(model_obj.dir.model_path)
            print("Loading model from: %s" % ckpt.model_checkpoint_path)
            tf.train.Saver().restore(session, ckpt.model_checkpoint_path)

            aspect_file = open('aspect_file.txt', 'w')

            aspect_emb_matrix, vocab_emb_matrix = session.run([model_obj.aspect_emb_matrix, model_obj.vocab_emb_marix])

            aspect_emb_norm = np.linalg.norm(aspect_emb_matrix + 1e-6, axis=-1, keepdims=True)
            normalized_aspect_emb = aspect_emb_matrix / aspect_emb_norm

            vocab_emb_norm = np.linalg.norm(vocab_emb_matrix + 1e-6, axis=-1, keepdims=True)
            normalized_vocab_emb = vocab_emb_matrix / vocab_emb_norm

            for each_aspect_idx in range(len(normalized_aspect_emb)):
                curr_aspect_emb = normalized_aspect_emb[each_aspect_idx]
                sim_values = np.dot(normalized_vocab_emb, curr_aspect_emb)
                ordered_index = np.argsort(sim_values)[::-1]
                word_list = [vocab_map.keys()[vocab_map.values().index(w)] for w in ordered_index[:100]]
                aspect_file.write('Aspect %d:\n' % each_aspect_idx)
                aspect_file.write(' '.join(word_list) + '\n\n')

            aspect_file.close()

    def getLength(self, fileName):
        print('Reading :', fileName)
        dataFile = open(fileName, 'r')
        count = 0
        for _ in dataFile:
            count += 1
        dataFile.close()
        return count, np.arange(count)

    def run_train(self, dict_obj):
        mode_train, mode_valid, mode_test = 'TR', 'VA', 'TE'

        # train object

        params_train = Params(mode=mode_train)
        dir_train = Directory(mode_train)
        params_train.num_instances, params_train.indices = self.getLength(dir_train.data_filename)

        # valid object

        params_valid = Params(mode=mode_valid)
        dir_valid = Directory(mode_valid)
        params_valid.num_instances, params_valid.indices = self.getLength(dir_valid.data_filename)

        # test object
        #
        # params_test = Params(mode=mode_test)
        # dir_test = Directory(mode_test)
        # params_test.num_instances, params_test.indices = self.getLength(dir_test.data_filename)

        random.seed(1234)
        if (params_train.enable_shuffle):
            random.shuffle(params_train.indices)
            random.shuffle(params_valid.indices)

        min_loss = sys.float_info.max

        word_emb_path = dir_train.word_embedding
        word_emb_matrix = np.float32(np.genfromtxt(word_emb_path, delimiter=' '))
        params_train.VOCAB_SIZE = params_valid.VOCAB_SIZE = len(word_emb_matrix)

        logger.info('***** INITIALIZING TF GRAPH *****')

        with tf.Graph().as_default(), tf.Session() as session:
            # train_writer = tf.summary.FileWriter(dir_train.log_path + '/train', session.graph)
            # test_writer = tf.summary.FileWriter(dir_train.log_path + '/test')

            # random_normal_initializer = tf.random_normal_initializer()
            # random_uniform_initializer = tf.random_uniform_initializer(-params_train.init_scale, params_train.init_scale)
            xavier_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)

            # with tf.name_scope('train'):
            with tf.variable_scope("model", reuse=None, initializer=xavier_initializer):
                train_obj = Model(params_train, dir_train)

            # with tf.name_scope('valid'):
            with tf.variable_scope("model", reuse=True, initializer=xavier_initializer):
                valid_obj = Model(params_valid, dir_valid)

            # with tf.variable_scope("model", reuse=True, initializer=xavier_initializer):
            #     test_obj = Model(params_test, dir_test)

            if not params_train.enable_checkpoint:
                session.run(tf.global_variables_initializer())

            if params_train.enable_checkpoint:
                ckpt = tf.train.get_checkpoint_state(dir_train.model_path)
                if ckpt and ckpt.model_checkpoint_path:
                    print("Loading model from: %s" % ckpt.model_checkpoint_path)
                    tf.train.Saver().restore(session, ckpt.model_checkpoint_path)
            elif (not params_train.use_random_initializer):
                self.assign_word_emb_matrix(session, train_obj, word_emb_matrix)
                self.assign_aspect_emb_matrix(session, params_train, word_emb_matrix, train_obj)

            model_saver = tf.train.Saver()
            model_saver.save(session, save_path=dir_train.model_path + dir_train.model_name, latest_filename=dir_train.latest_checkpoint)
            print('==== Model saved! ====')

            logger.info('**** TF GRAPH INITIALIZED ****')

            # train_writer.add_graph(tf.get_default_graph())

            start_time = time.time()
            reader = DataReader()
            for i in range(params_train.max_max_epoch):
                lr_decay = params_train.lr_decay ** max(i - params_train.max_epoch, 0.0)
                # train_obj.assign_lr(session, params_train.learning_rate * lr_decay)

                # print(params_train.learning_rate * lr_decay)
                self.get_aspect_words(train_obj, dict_obj.word_dict)
                print('\n++++++++=========+++++++\n')
                lr = params_train.learning_rate * lr_decay
                logger.info("Epoch: %d Learning rate: %.5f" % (i + 1, lr))
                train_loss, _, summary = self.run_epoch(session, min_loss, train_obj, dict_obj, i, reader, lr, verbose=True)
                print("Epoch: %d Train loss: %.4f" % (i + 1, train_loss))

                valid_loss, curr_loss, summary = self.run_epoch(session, min_loss, valid_obj, dict_obj, i, reader)
                if (curr_loss < min_loss):
                    min_loss = curr_loss

                print("Epoch: %d Valid loss: %.4f" % (i + 1, valid_loss))

                # test_loss, _, _ = self.run_epoch(session, min_loss, test_obj, dict_obj, i, reader)
                # print("Epoch: %d Test loss: %.4f" % (i + 1, test_loss))

                curr_time = time.time()
                print('1 epoch run takes ' + str(((curr_time - start_time) / (i + 1)) / 60) + ' minutes.')

                # train_writer.close()
                # test_writer.close()
            self.get_aspect_words(train_obj, dict_obj.word_dict)

    def assign_aspect_emb_matrix(self, session, params_train, word_emb_matrix, train_obj):
        np.random.seed(1234)
        km = KMeans(n_clusters=params_train.ASPECT_NUM)
        km.fit(word_emb_matrix)
        clusters = km.cluster_centers_
        norm_aspect_matrix = clusters / np.linalg.norm(clusters, axis=-1, keepdims=True)
        session.run(tf.assign(train_obj.aspect_emb_matrix, norm_aspect_matrix, name='aspect_emb_matrix'))

    def assign_word_emb_matrix(self, session, train_obj, word_emb_matrix):
        norm_emb_matrix = word_emb_matrix / np.linalg.norm(word_emb_matrix, axis=-1, keepdims=True)
        session.run(tf.assign(train_obj.vocab_emb_marix, norm_emb_matrix, name="word_embedding_matrix"))
