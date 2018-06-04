from global_module.settings_module import Params, Dictionary, Directory
import cPickle
import csv
import pickle
import random


class VocabBuilder:
    def __init__(self):
        self.params = Params('TR')

    def create_vocab_dict(self, input_filename):
        word_dict = {}
        word_count = {}
        word_counter = 2
        max_sequence_length = 0

        input_file = open(input_filename, 'r').readlines()

        for each_line in input_file:
            if self.params.all_lowercase:
                each_line = each_line.lower()
            tokenized_string = each_line.strip().split()

            for each_token in tokenized_string:
                if each_token != 'unk':
                    if each_token not in word_count:
                        word_count[each_token] = 1
                    else:
                        word_count[each_token] = 1 + word_count[each_token]

            line_len = len(tokenized_string)
            if (line_len > max_sequence_length):
                max_sequence_length = line_len

        for each_token in word_count:
            if word_count[each_token] >= self.params.sampling_threshold:
                word_dict[each_token] = word_counter
                word_counter += 1

        word_vocab = open(Directory('TR').word_vocab_dict, 'wb')
        pickle.dump(word_dict, word_vocab, protocol=cPickle.HIGHEST_PROTOCOL)
        word_vocab.close()

        print('Reading Completed \n ========================== '
              '\n Unique tokens: excluding padding and unkown words %d '
              '\n Max. sequence length: %d'
              '\n ==========================\n'
              % (word_counter - 2, max_sequence_length))

    def extract_glove_vectors(self, word_vocab_file, glove_file):
        glove_vocab_dict = cPickle.load(open(glove_file, 'rb'))
        word_vocab_dict = cPickle.load(open(word_vocab_file, 'rb'))

        length_word_vector = self.params.EMB_DIM

        glove_present_training_word_vocab_dict = {}
        glove_present_training_word_counter = 1
        glove_present_word_vector_dict = {}

        freq_thres = len(glove_vocab_dict)

        if 'unk' not in glove_vocab_dict and 'UNK' not in glove_vocab_dict:
            glove_present_training_word_vocab_dict['unk'] = 1
            vec_str = ''
            for i in range(length_word_vector):
                vec_str += str(round(random.uniform(-0.1, 0.1), 6)) + ' '
            glove_present_word_vector_dict[1] = vec_str.strip()
            glove_present_training_word_counter = 2

        for key, value in glove_vocab_dict.items()[:freq_thres]:
            # if key == 'UNK':
            #     key = key.lower()
            vec = glove_vocab_dict.get(key)
            if self.params.all_lowercase:
                key = key.lower()
                if key in glove_vocab_dict:
                    vec = glove_vocab_dict.get(key)

            if key not in glove_present_training_word_vocab_dict:
                glove_present_training_word_vocab_dict[key] = glove_present_training_word_counter
                glove_present_word_vector_dict[glove_present_training_word_counter] = vec
                glove_present_training_word_counter += 1

        #to use training words not present in top frequent words of glove
        for key, value in word_vocab_dict.items():
            if key not in glove_present_training_word_vocab_dict and key in glove_vocab_dict:
                glove_present_training_word_vocab_dict[key] = glove_present_training_word_counter
                glove_present_word_vector_dict[glove_present_training_word_counter] = glove_vocab_dict[key]
                glove_present_training_word_counter += 1

            # to use unknown words
            elif self.params.use_unknown_word:
                if self.params.all_lowercase:
                    key = key.lower()
                glove_present_training_word_vocab_dict[key] = glove_present_training_word_counter
                vec_str = ''
                for i in range(length_word_vector):
                    vec_str += str(round(random.uniform(-0.1, 0.1), 6)) + ' '
                glove_present_word_vector_dict[glove_present_training_word_counter] = vec_str.strip()
                glove_present_training_word_counter += 1

        word_vector_file = open(Directory('TR').word_embedding, 'w')
        writer = csv.writer(word_vector_file)
        string = ''
        for i in range(length_word_vector):
            string += '0.000001 '
        word_vector_file.write(string.rstrip(' ') + '\n')
        # word_vector_file.write(string.rstrip(' ') + '\n') # zeros vector (id 1)
        for key, value in glove_present_word_vector_dict.items():
            writer.writerow([value])

        # op_file = open('abc.txt', 'w')
        # for each_word in glove_present_training_word_vocab_dict:
        #     op_file.write(each_word + ' ' + str(glove_present_training_word_vocab_dict[each_word]) + '\n')
        # op_file.close()

        glove_present_training_word_vocab = open(Directory('TR').glove_present_training_word_vocab, 'wb')
        pickle.dump(glove_present_training_word_vocab_dict, glove_present_training_word_vocab, protocol=cPickle.HIGHEST_PROTOCOL)

        print(glove_present_training_word_vocab_dict)

        print('Glove_present_unique_training_tokens, Total unique tokens, Glove token size')
        print(len(glove_present_training_word_vocab_dict), len(word_vocab_dict), len(glove_vocab_dict))

        word_vector_file.close()

        print('\nVocab Size:')
        # print(len(glove_present_word_vector_dict)+2)
        print(len(glove_present_word_vector_dict) + 1)

        glove_present_training_word_vocab.close()
        # return(len(glove_present_word_vector_dict)+2)
        return (len(glove_present_word_vector_dict) + 1)