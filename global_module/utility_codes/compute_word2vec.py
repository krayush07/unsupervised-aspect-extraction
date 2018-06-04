import gensim
import codecs
from global_module.settings_module import Directory

class MySentences(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in codecs.open(self.filename, 'r', 'utf-8'):
            yield line.split()


def main(domain):
    data_dir = Directory('TR').data_path
    source = data_dir + '/%s/pre_processed_restaurant/train.txt' % (domain)
    model_file = data_dir + '/%s/pre_processed_restaurant/w2v_embedding_new' % (domain)
    sentences = MySentences(source)
    model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=10, workers=4)
    model.save(model_file)


print 'Pre-training word embeddings ...'
main('restaurant')
# main('beer')