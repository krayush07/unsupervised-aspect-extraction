import gensim
import pickle, cPickle
import collections
import numpy as np
from global_module.settings_module import Directory


def convert(emb_file):
    # model = gensim.models.KeyedVectors.load_word2vec_format(emb_file, binary=True, unicode_errors='ignore')
    model = gensim.models.Word2Vec.load(emb_file)
    output_emb_file = open(emb_file + '_dict.pkl', 'wb')
    vocab_dict = collections.OrderedDict()

    for word in model.wv.index2word:
        word_vec = model[word]
        # print word, model.wv.most_similar(positive=word)
        word_vec = np.array2string(word_vec)
        word_vec = word_vec.replace('[', '')
        word_vec = word_vec.replace(']', '')
        vocab_dict[word] = (' '.join(word_vec.split())).strip()

    pickle.dump(vocab_dict, output_emb_file, protocol=cPickle.HIGHEST_PROTOCOL)

base_dir = Directory('TR').data_path
convert(base_dir + '/restaurant/pre_processed_restaurant/w2v_embedding')