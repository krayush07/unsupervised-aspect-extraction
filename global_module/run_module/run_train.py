from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from global_module.implementation_module import Train
from global_module.settings_module import set_dict
from global_module.pre_processing_module import VocabBuilder
from global_module.settings_module import Directory

def load_dictionary():
    """
    Utility function to load training vocab files
    :return:
    """
    return set_dict.Dictionary()


def call_train(dict_obj):
    """
    Utility function to execute main training module
    :param dict_obj: dictionary object
    :return: None
    """
    Train().run_train(dict_obj)
    return


def train_util():
    """
    Utility function to execute the training pipeline
    :return: None
    """

    # To use training set only
    # build_sampled_training_file.util()
    # build_word_vocab.util()

    # To use complete glove file + training set

    dir_obj = Directory('TR')
    vocab_builder = VocabBuilder()
    vocab_builder.create_vocab_dict(dir_obj.data_filename)
    vocab_builder.extract_glove_vectors(dir_obj.word_vocab_dict, dir_obj.glove_path)

    dict_obj = load_dictionary()
    call_train(dict_obj)
    return None


def main():
    """
    Starting module for CLSTM testing
    :return:
    """
    print('STARTING TRAINING')
    train_util()
    return None


if __name__ == '__main__':
    main()
