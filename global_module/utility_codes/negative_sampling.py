import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy import spatial
from global_module.settings_module import Directory

def get_cosine_similarity(arr1, arr2, delimiter=None):
    # arr1 = np.array(map(np.float32, str1.split(delimiter)), dtype=np.float32)
    # arr2 = np.array(map(np.float32, str2.split(delimiter)), dtype=np.float32)

    arr_size = len(arr1)
    zero_count_arr1 = arr_size - np.count_nonzero(arr1)

    if (zero_count_arr1 == arr_size):
        zero_count_arr2 = arr_size - np.count_nonzero(arr2)
        if (zero_count_arr2 == arr_size):
            return 1.0
        else:
            return 0.0
    else:
        zero_count_arr2 = arr_size - np.count_nonzero(arr2)
        if (zero_count_arr2 == arr_size):
            return 0.0

    cosine_val = 1 - spatial.distance.cosine(arr1, arr2)
    # if (cosine_val < 0):
    #     cosine_val = 0.0
    # elif (str(cosine_val) == 'nan'):
    #     cosine_val = 1.0

    if (str(cosine_val) == 'nan'):
        cosine_val = 0.0

    return cosine_val

def random_sampling(train_file, input_file):
    random.seed(789)

    train_lines = open(train_file, 'r').readlines()
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=5)
    train_features = vectorizer.fit_transform([utt for utt in train_lines])

    lines = open(input_file, 'r').readlines()
    idx_features = vectorizer.transform([utt for utt in lines])


    output_file = open(input_file + '_sampled.txt', 'w')

    for idx, each_line in enumerate(lines):
        counter = 0
        while counter < 20:
            sampled_idx = random.randint(0, len(lines)-1)
            if sampled_idx != idx:
                # cosine_val = get_cosine_similarity(idx_features[idx].todense(), idx_features[sampled_idx].todense())
                # if cosine_val <= 0.8:
                    output_file.write(each_line.strip() + '\t' + lines[sampled_idx].strip() + '\n')
                    counter += 1
    output_file.close()

data_dir = Directory('TR').data_path
random_sampling(data_dir + '/restaurant/authors_preprocessed/tokenized_train.txt',
                data_dir + '/restaurant/authors_preprocessed/tokenized_train.txt')