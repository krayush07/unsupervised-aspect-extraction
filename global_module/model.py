from global_module.settings_module import Directory, Params
from global_module.implementation_module import Encoder, Decoder
import tensorflow as tf


class Model:
    def __init__(self, params, dir_obj):
        self.params = params
        self.dir = dir_obj
        self.eps = tf.constant(1e-6, dtype=tf.float32)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.init_pipeline()

    def init_pipeline(self):
        self.create_placeholders()
        self.create_emb_matrix()
        self.embedding_lookup()
        self.pos_avg_sentence_rep = self.compute_avg_sentence_representation(self.inp_emb_mat, self.inp_seq_len)
        self.neg_avg_sentence_rep = self.compute_avg_sentence_representation(self.neg_inp_emb_mat, self.neg_seq_len)
        self.attention_weights, self.encoded_rep, self.pre_attention_scores, self.filter_layer_representation = self.encoder.encode(self.inp_emb_mat, self.params.EMB_DIM, self.pos_avg_sentence_rep)
        self.aspect_weights, self.decoded_rep = self.decoder.decode(self.aspect_emb_matrix, self.params.ASPECT_NUM, self.encoded_rep)
        self.compute_loss()
        self.train()

    def create_placeholders(self):
        self.input_utterance = tf.placeholder(dtype=tf.int32,
                                              shape=[None, self.params.MAX_SEQ_LEN],
                                              name='inp_utt_placeholder')

        self.negative_utterance = tf.placeholder(dtype=tf.int32,
                                                 shape=[None, self.params.MAX_SEQ_LEN],
                                                 name='neg_utt_placeholder')

        self.inp_seq_len = tf.placeholder(dtype=tf.int32,
                                          shape=[None],
                                          name='inp_seq_len_placeholder')

        self.neg_seq_len = tf.placeholder(dtype=tf.int32,
                                          shape=[None],
                                          name='neg_seq_len_placeholder')

        self.keep_prob = tf.placeholder(dtype=tf.float32,
                                        shape=(),
                                        name='dropout_placeholder')

        self.lr = tf.placeholder(dtype=tf.float32,
                                 shape=(),
                                 name='lr_placeholder')

    def create_emb_matrix(self):
        self.vocab_emb_marix = tf.get_variable(name='vocab_emb',
                                               shape=[self.params.VOCAB_SIZE, self.params.EMB_DIM],
                                               dtype=tf.float32,
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               trainable=self.params.IS_WORD_TRAINABLE)

        self.aspect_emb_matrix = tf.get_variable(name='aspect_emb',
                                                 shape=[self.params.ASPECT_NUM, self.params.EMB_DIM],
                                                 dtype=tf.float32,
                                                 initializer=tf.contrib.layers.xavier_initializer(),
                                                 trainable=True)

    def embedding_lookup(self):
        self.inp_emb_mat = tf.nn.embedding_lookup(self.vocab_emb_marix, self.input_utterance)
        self.neg_inp_emb_mat = tf.nn.embedding_lookup(self.vocab_emb_marix, self.negative_utterance)

    def compute_avg_sentence_representation(self, input_matrix, seq_len):
        sum_representation = tf.reduce_sum(input_matrix, axis=1)
        avg_representation = tf.divide(sum_representation, tf.expand_dims(tf.cast(seq_len, dtype=tf.float32), axis=-1))
        return avg_representation

    def compute_regularized_loss(self):
        aspect_emb_norm = tf.norm(self.aspect_emb_matrix + self.eps, axis=-1, keepdims=True)
        normalized_aspect_emb = self.aspect_emb_matrix / aspect_emb_norm
        identity_matrix = tf.eye(num_rows=self.params.ASPECT_NUM, dtype=tf.float32, name='identity_matrix')
        self.aspect_dot = tf.matmul(normalized_aspect_emb, normalized_aspect_emb, transpose_b=True)
        regularized_matrix = tf.square(tf.matmul(normalized_aspect_emb, normalized_aspect_emb, transpose_b=True) - identity_matrix)
        return self.params.REG_CONSTANT * tf.reduce_sum(regularized_matrix)

    def compute_loss(self):
        self.regularized_loss = self.compute_regularized_loss()
        r_s = self.decoded_rep / tf.norm(self.decoded_rep + self.eps, axis=-1, keepdims=True)
        z_s = self.encoded_rep / tf.norm(self.encoded_rep + self.eps, axis=-1, keepdims=True)
        z_n = self.neg_avg_sentence_rep / tf.norm(self.neg_avg_sentence_rep + self.eps, axis=-1, keepdims=True)
        pos_inner_product = tf.multiply(z_s, r_s, name='pos_inner_prod')
        neg_inner_product = tf.multiply(z_n, r_s, name='neg_inner_prod')
        self.pos_loss = tf.reduce_sum(pos_inner_product, axis=1)
        self.neg_loss = tf.reduce_sum(neg_inner_product, axis=1)
        # self.pos_indiv_loss = tf.square(tf.abs(1. - self.pos_loss))
        # self.neg_indiv_loss = tf.square(tf.abs(1. + self.neg_loss))
        self.total_loss = self.regularized_loss + tf.reduce_sum(tf.nn.relu(tf.constant(1.25, dtype=tf.float32)
                                                                           - self.pos_loss
                                                                           + self.neg_loss)) # + tf.reduce_sum(self.pos_indiv_loss) + tf.reduce_sum(self.neg_indiv_loss)
        print 'loss computed'

    def train(self):
        global optimizer
        with tf.variable_scope('optimize'):
            if self.params.mode != 'TR':
                self.train_op = tf.no_op()
                return

            learning_rate = self.lr

            trainable_tvars = tf.trainable_variables()
            grads = tf.gradients(self.total_loss, trainable_tvars)
            grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.params.max_grad_norm)
            grad_var_pairs = zip(grads, trainable_tvars)

            if self.params.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name='sgd')
            elif self.params.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='adam')
            elif self.params.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, epsilon=1e-6, name='adadelta')

            self.train_op = optimizer.apply_gradients(grad_var_pairs, name='apply_grad')


def main():
    model_obj = Model(Params('TR'), Directory('TR'))


if __name__ == '__main__':
    main()
