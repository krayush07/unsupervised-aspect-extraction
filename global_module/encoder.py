import tensorflow as tf


class Encoder:
    def encode(self, input_matrix, emb_dim, avg_sent_rep):
        attention_weights, pre_attention_scores, filter_layer_representation = self.compute_attention_weights(input_matrix, emb_dim, avg_sent_rep)
        encoded_rep = self.encode_input(input_matrix, attention_weights)
        return attention_weights, encoded_rep, pre_attention_scores, filter_layer_representation

    def compute_attention_weights(self, input_matrix, emb_dim, avg_sent_rep):
        init = tf.random_uniform_initializer(-1.0, 1.0)
        filter_layer_representation = tf.layers.dense(inputs=input_matrix,
                                                      units=emb_dim,
                                                      use_bias=False,
                                                      kernel_initializer=init,
                                                      activation=tf.nn.leaky_relu,
                                                      name='filter_layer')

        pre_attention_scores = tf.squeeze(tf.matmul(a=filter_layer_representation,
                                                    b=tf.expand_dims(avg_sent_rep, axis=2)),
                                          axis=2)

        # pre_attention_scores = tf.nn.leaky_relu(pre_attention_scores)

        attention_weights = tf.nn.softmax(pre_attention_scores, name='attention_weights')
        return attention_weights, pre_attention_scores, filter_layer_representation

    def encode_input(self, inp_emb_matrix, attention_weights):
        weighted_input = tf.matmul(inp_emb_matrix, tf.expand_dims(attention_weights, axis=-1), transpose_a=True)
        return tf.squeeze(weighted_input, axis=2)
