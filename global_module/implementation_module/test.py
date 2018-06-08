import tensorflow as tf
import numpy as np
import sys
import time

from global_module.settings_module import Directory
from global_module.implementation_module import DataReader, Model
from global_module.settings_module import Directory, Dictionary, Params
from tensorflow.python.saved_model import builder as saved_model_builder
import logging
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class Test:
    def run_epoch(self, session, model_obj, dict_obj, reader, cluster):
        combined_loss = -1.0
        epoch_combined_loss = 0.0
        total_instances = 0.0
        step = 0
        print('\nrun epoch')

        params = model_obj.params
        dir_obj = model_obj.dir
        data_filename = dir_obj.data_filename
        label_filename = dir_obj.label_filename

        data_lines = open(data_filename).readlines()
        label_lines = open(label_filename).readlines()

        output_file = open(model_obj.dir.output_path + '/test_predictions.txt', 'w')

        gold_label = []
        predicted_label = []

        for step, (pos_seq_arr, pos_seq_len) \
                in enumerate(reader.data_iterator(params, data_filename, model_obj.params.indices, dict_obj)):
            feed_dict = {}

            if len(pos_seq_arr) != params.batch_size:
                print 'WRONG'

            if len(pos_seq_len) != params.batch_size:
                print 'Wrong'

            feed_dict[model_obj.input_utterance] = pos_seq_arr
            feed_dict[model_obj.inp_seq_len] = pos_seq_len
            feed_dict[model_obj.keep_prob] = params.keep_prob

            try:
                pos_loss, aspect_weights, _ = session.run([model_obj.pos_loss,
                                                          model_obj.aspect_weights,
                                                          model_obj.train_op],
                                                        feed_dict = feed_dict)

                for idx, each_aspect_weight in enumerate(aspect_weights):
                    predicted_aspect = np.argmax(each_aspect_weight)
                    output_file.write(cluster[predicted_aspect] + '\t' + data_lines[step * len(pos_seq_len) + idx].strip() + '\n')
                    gold_label.append(label_lines[step * len(pos_seq_len) + idx].strip().lower())
                    predicted_label.append(cluster[predicted_aspect])

                total_instances += params.batch_size
                epoch_combined_loss += np.sum(pos_loss)
            except:
                logger.info('Error detected')

        print(classification_report(gold_label, predicted_label,
                                    ['food', 'staff', 'ambience', 'anecdote', 'price', 'misc'], digits=3))

        mean_loss = epoch_combined_loss / step
        print 'CE loss: %.4f' % (mean_loss)
        output_file.close()
        return mean_loss


    def getLength(self, fileName):
        print('Reading :', fileName)
        dataFile = open(fileName, 'r')
        count = 0
        for _ in dataFile:
            count += 1
        dataFile.close()
        return count, np.arange(count)


    def init_test(self):
        mode_train, mode_valid, mode_test = 'TR', 'VA', 'TE'

        # test object

        params_test = Params(mode=mode_test)
        dir_test = Directory(mode_test)
        params_test.num_instances, params_test.indices = self.getLength(dir_test.data_filename)

        min_loss = sys.float_info.max

        word_emb_path = dir_test.word_embedding
        word_emb_matrix = np.float32(np.genfromtxt(word_emb_path, delimiter=' '))
        params_test.VOCAB_SIZE = len(word_emb_matrix)
        params_test.batch_size = 10

        print('***** INITIALIZING TF GRAPH *****')

        session = tf.Session()

        # with tf.name_scope('train'):
        with tf.variable_scope("model", reuse=None):
            test_obj = Model(params_test, dir_test)

        model_saver = tf.train.Saver()
        print('Loading model ...')
        model_saver.restore(session, dir_test.test_model)

        # builder = saved_model_builder.SavedModelBuilder(set_dir.Directory('TE').test_model)
        # builder.add_meta_graph_and_variables(session, ['serve'], signature_def_map=None)
        # builder.save()
        # return

        print('**** MODEL LOADED ****\n')

        return session, test_obj


    def run_test(self, session, test_obj, dict_obj, reader, cluster):
        start_time = time.time()

        print("Starting test computation\n")
        test_loss = self.run_epoch(session, test_obj, dict_obj, reader, cluster)

        curr_time = time.time()
        print('1 epoch run takes ' + str(((curr_time - start_time) / 60)) + ' minutes.')

# def main():
#     session, test_obj = init_test()
#     dict_obj = set_dict.Dictionary()
#     run_test(session, test_obj, dict_obj)
#
#
# if __name__ == "__main__":
#     main()
