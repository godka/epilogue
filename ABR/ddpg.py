"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import gym
import time
import tflearn


#####################  hyper parameters  ####################

LR_A = 1e-4    # learning rate for actor
LR_C = 1e-3    # learning rate for critic
GAMMA = 0.99     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

S_INFO = 6
S_LEN = 8  # take how many frames in the past
A_DIM = 6
FEATURE_NUM = 64
###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        #print s.shape, a.shape, r.shape, s_.shape
        s = np.reshape(s, (self.s_dim))
        a = np.reshape(a, (self.a_dim))
        s_ = np.reshape(s_, (self.s_dim))
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

#def create_network(self):
    #    inputs = tflearn.input_data(shape=[None, S_INFO, S_LEN], name='input')
    #    split_0 = tflearn.fully_connected(
    #        inputs[:, 0:1, :], FEATURE_NUM, activation='relu')
    #    split_1 = tflearn.fully_connected(
    #        inputs[:, 1:2, :], FEATURE_NUM, activation='relu')
    #    split_2 = tflearn.conv_1d(
    #         inputs[:, 2:3, :], FEATURE_NUM, 4, activation='relu')
    #    split_3 = tflearn.conv_1d(
    #        inputs[:, 3:4, :], FEATURE_NUM, 4, activation='relu')
    #    split_4 = tflearn.conv_1d(
    #         inputs[:, 4:5, :A_DIM], FEATURE_NUM, 4, activation='relu')
    #    split_5 = tflearn.fully_connected(
    #        inputs[:, 5:6, :], FEATURE_NUM, activation='relu')
#
#        split_2_flat = tflearn.flatten(split_2)
#        split_3_flat = tflearn.flatten(split_3)
#        split_4_flat = tflearn.flatten(split_4)
#
#        
#        net = tf.stack([split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], axis=1)
#        net = tflearn.fully_connected(net, FEATURE_NUM, activation='relu')
#        out = tflearn.fully_connected(net, A_DIM, activation='softmax', name='predictions')
        # merge_net, alphas = self.attention(net, FEATURE_NUM)
        # dense_net_0 = self.nac(merge_net, FEATURE_NUM)
        # out = self.nac(dense_net_0, A_DIM)
        # out = tf.nn.softmax(out)
#        return inputs, out, None

    def _build_a(self, s, scope, trainable):
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

            
            net = tf.stack([split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], axis=1)
            net = tflearn.fully_connected(net, FEATURE_NUM, activation='relu', trainable=trainable)
            a = tflearn.fully_connected(net, self.a_dim, activation='sigmoid', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
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
            
            split_6 = tflearn.fully_connected(
                a, FEATURE_NUM, activation='relu', trainable=trainable)

            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)

            
            net = tf.stack([split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5, split_6], axis=1)
            net = tflearn.fully_connected(net, FEATURE_NUM, activation='relu', trainable=trainable)
            net = tflearn.fully_connected(net, 1, activation='linear', trainable=trainable)
            return net
