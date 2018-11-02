import tensorflow as tf
import tflearn
import os
import numpy as np
S_INFO = 6
S_LEN = 8  # take how many frames in the past
A_DIM = 6
FEATURE_NUM = 64


class bbaplus(object):
    def __init__(self, sess, filename="models/model.ckpt"):
        #gpu_options = tf.GPUOptions(allow_growth=True)
        #self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = sess
        self.S = tf.placeholder(tf.float32, [None, S_INFO * S_LEN], name = 's')
        self.res, self.alphas = self.create_network(self.S)
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.sess, filename)
        print 'model restored.'
        
    def attention(self, inputs, attention_size, trainable=True):
        # the length of sequences processed in the antecedent RNN layer
        if isinstance(inputs, list):
            inputs = tf.stack(inputs, axis=1)
        sequence_length = inputs.get_shape()[1].value
        hidden_size = inputs.get_shape()[2].value  # hidden size of the RNN layer

        # Attention mechanism
        W_omega = tf.Variable(tf.random_normal(
            [hidden_size, attention_size], stddev=0.1), trainable=trainable)
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1), trainable=trainable)
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1), trainable=trainable)

        v = tf.tanh(tf.matmul(tf.reshape(
            inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
        vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
        exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
        alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
        # Output of Bi-RNN is reduced with attention vector
        output = tf.reduce_sum(
            inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)

        return output, alphas

    def nac(self, input_layer, num_outputs, trainable=True):
        shape = (int(input_layer.shape[-1]), num_outputs)

        # define variables
        W_hat = tf.Variable(tf.truncated_normal(shape, stddev=0.02), trainable=trainable)
        M_hat = tf.Variable(tf.truncated_normal(shape, stddev=0.02), trainable=trainable)
        # operations according to paper
        W = tf.tanh(W_hat) * tf.sigmoid(M_hat)
        a = tf.matmul(input_layer, W)
        return a

    def create_network(self, s, scope='Actor/eval', trainable=True):
        with tf.variable_scope(scope):
            inputs = tf.reshape(s, [-1, S_INFO, S_LEN])
            split_0 = tflearn.fully_connected(
                inputs[:, 0:1, :], FEATURE_NUM, activation='relu', trainable=trainable)
            split_1 = tflearn.fully_connected(
                inputs[:, 1:2, :], FEATURE_NUM, activation='relu', trainable=trainable)
            split_2 = tflearn.conv_1d(
                inputs[:, 2:3, :], FEATURE_NUM, 4, activation='relu', trainable=trainable)
            split_3 = tflearn.conv_1d(
                inputs[:, 3:4, :], FEATURE_NUM, 4, activation='relu', trainable=trainable)
            split_4 = tflearn.conv_1d(
                inputs[:, 4:5, :A_DIM], FEATURE_NUM, 4, activation='relu', trainable=trainable)
            split_5 = tflearn.fully_connected(
                inputs[:, 5:6, :], FEATURE_NUM, activation='relu', trainable=trainable)

            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)

            net = tf.stack([split_0, split_1, split_2_flat,
                            split_3_flat, split_4_flat, split_5], axis=1)
            net = tflearn.fully_connected(net, FEATURE_NUM, activation='relu', trainable=trainable)
            merge_net, alphas = self.attention(net, FEATURE_NUM)
            dense_net_0 = self.nac(merge_net, FEATURE_NUM, trainable=trainable)
            out = self.nac(dense_net_0, 2, trainable=trainable)
            out = tf.nn.sigmoid(out, trainable=trainable)
            #a = tflearn.fully_connected(
            #    net, 2, activation='sigmoid', trainable=trainable)
            return tf.multiply(out, 60., name='scaled_a'), alphas

    def predict(self, state):
        _state = np.reshape(state, (-1, S_INFO * S_LEN))
        _res = self.sess.run(self.res, feed_dict={self.S: _state})
        return _res[0]

    def print_name(self):
        for v in tf.global_variables():
            print(v)

    def print_checkpoint_file(self, filename = "models/model.ckpt"):
        checkpoint_file = filename
        try:
            reader = tf.train.NewCheckpointReader(checkpoint_file)
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in sorted(var_to_shape_map):
                print "tensor_name: ", key
        except Exception as e:  # pylint: disable=broad-except
            print str(e)
