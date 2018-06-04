import tensorflow as tf


class Decoder:
    def decode(self, aspect_emb_matrix, aspect_num, encoded_rep):
        aspect_weights = self.compute_aspect_weights(encoded_rep, aspect_num)
        decoded_rep = self.decode_input(aspect_emb_matrix, aspect_weights)
        return aspect_weights, decoded_rep

    def compute_aspect_weights(self, encoded_rep, aspect_num):
        reduced_representation = tf.layers.dense(inputs=encoded_rep,
                                                 units=aspect_num,
                                                 use_bias=True,
                                                 name='reduction_layer')
        return tf.nn.softmax(reduced_representation, name='aspect_weights')

    def decode_input(self, aspect_emb_matrix, aspect_weights):
        return tf.matmul(a=aspect_weights,
                         b=aspect_emb_matrix,
                         name='decoded_rep')