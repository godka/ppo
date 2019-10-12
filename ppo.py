import math
import numpy as np
import tensorflow as tf
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import tflearn

FEATURE_NUM = 128
ACTION_EPS = 1e-4
GAMMA = 0.99
# PPO2
EPS = 0.2

class Network():
    def CreateNetwork(self, inputs):
        with tf.variable_scope('actor'):
            pi = tflearn.fully_connected(self.CreateCore(inputs), self.a_dim, activation='softmax')
            value = tflearn.fully_connected(self.CreateCore(inputs), 1, activation='linear')
            return pi, value

    def CreateCore(self, inputs):
        net = tflearn.fully_connected(inputs, 32, activation='relu')
        net = tflearn.fully_connected(net, 16, activation='relu')
        net = tflearn.fully_connected(net, 8, activation='relu')
        return net

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })

    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.training_epo = 10
        self.quality = 0
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.sess = sess
        self.R = tf.placeholder(tf.float32, [None, 1])
        self.inputs = tf.placeholder(tf.float32, [None, self.s_dim])
        self.old_pi = tf.placeholder(tf.float32, [None, self.a_dim])
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])
        self.entropy_weight = tf.placeholder(tf.float32)

        self.pi, self.val = self.CreateNetwork(inputs=self.inputs)
        self.real_out = tf.clip_by_value(self.pi, ACTION_EPS, 1. - ACTION_EPS)
        self.entropy = tf.multiply(self.real_out, tf.log(self.real_out))
        self.adv = self.R - tf.stop_gradient(self.val)
        
        self.ratio = tf.reduce_sum(tf.multiply(self.real_out, self.acts), reduction_indices=1, keepdims=True) / \
                tf.reduce_sum(tf.multiply(self.old_pi, self.acts), reduction_indices=1, keepdims=True)

        self.ppo2loss = tf.minimum(self.ratio * self.adv, 
                            tf.clip_by_value(self.ratio, 1 - EPS, 1 + EPS) * self.adv
                        )
        
        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(
                self.network_params[idx].assign(param))
        
        self.loss =  - tf.reduce_sum(self.ppo2loss) \
            + self.entropy_weight * tf.reduce_sum(self.entropy)
        
        self.optimize = tf.train.AdamOptimizer(self.lr_rate).minimize(self.loss)

        self.val_loss = 0.5 * tflearn.mean_square(self.val, self.R)
        self.val_optimize = tf.train.AdamOptimizer(self.lr_rate * 10.).minimize(self.val_loss)
    
    def predict(self, input):
        action = self.sess.run(self.real_out, feed_dict={
            self.inputs: input
        })
        return action[0]

    def get_entropy(self, step):
        return 0.1

    def train(self, s_batch, a_batch, p_batch, v_batch, epoch, batch_size = 32):
        # shuffle is all you need
        s_batch, a_batch, p_batch, v_batch = \
            tflearn.data_utils.shuffle(s_batch, a_batch, p_batch, v_batch)
        # mini_batch
        i, train_len = 0, s_batch.shape[0]
        while train_len > 0:
            _batch_size = np.minimum(batch_size, train_len)
            self.sess.run([self.optimize, self.val_optimize], feed_dict={
                self.inputs: s_batch[i:i+_batch_size],
                self.acts: a_batch[i:i+_batch_size],
                self.R: v_batch[i:i+_batch_size], 
                self.old_pi: p_batch[i:i+_batch_size],
                self.entropy_weight: self.get_entropy(epoch)
            })
            train_len -= _batch_size
            i += _batch_size

    def compute_v(self, s_batch, a_batch, r_batch, terminal):
        ba_size = len(s_batch)
        R_batch = np.zeros([len(r_batch), 1])

        v_batch = self.sess.run(self.val, feed_dict={
            self.inputs: s_batch
        })
        if terminal:
            R_batch[-1, 0] = 0  # terminal state
        else:    
            R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state
        for t in reversed(range(ba_size - 1)):
            R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

        td_batch = R_batch - v_batch
        return list(R_batch), list(td_batch)

    
    def compute_gae_v(self, s_batch, a_batch, r_batch, terminal, GAE_LAMBDA = 0.95):
        ba_size = len(s_batch)
        R_batch = np.zeros([len(r_batch), 1])
        mb_advs = np.zeros_like(R_batch)
        lastgaelam = 0.

        v_batch = self.sess.run(self.val, feed_dict={
            self.inputs: s_batch
        })
        if terminal:
            v_batch[-1, 0] = 0  # terminal state

        for t in reversed(range(ba_size - 1)):
            delta = r_batch[t] + GAMMA * v_batch[t+1] - v_batch[t] 
            mb_advs[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * lastgaelam 
        R_batch = mb_advs + v_batch
        
        return list(R_batch), list(mb_advs)