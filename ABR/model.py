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
        self.res = self.create_network(self.S)
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.sess, filename)
        print 'model restored.'

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
            net = tflearn.fully_connected(
                net, FEATURE_NUM, activation='relu', trainable=trainable)
            a = tflearn.fully_connected(
                net, 2, activation='sigmoid', trainable=trainable)
            return tf.multiply(a, 60., name='scaled_a')

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
