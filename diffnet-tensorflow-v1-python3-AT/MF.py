from __future__ import absolute_import
from __future__ import division
import os
import math
import numpy as np
from multiprocessing import Pool
from multiprocessing import cpu_count
import argparse
import logging
from time import time
from time import strftime
from time import localtime
import Dataset

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
_user_input = None
_item_input_pos = None
_batch_size = None
_index = None
_model = None
_sess = None
_dataset = None
_K = None
_feed_dict = None
_output = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run AMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='yelp',
                        help='Choose a dataset.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Evaluate per X epochs.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--dns', type=int, default=1,
                        help='number of negative sample for each positive in dns.')
    parser.add_argument('--reg', type=float, default=0,
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--reg_adv', type=float, default=1,
                        help='Regularization for adversarial loss')
    parser.add_argument('--restore', type=str, default=None,
                        help='The restore time_stamp for weights in \Pretrain')
    parser.add_argument('--ckpt', type=int, default=100,
                        help='Save the model per X epochs.')
    parser.add_argument('--task', nargs='?', default='',
                        help='Add the task name for launching experiments')
    parser.add_argument('--adv_epoch', type=int, default=0,
                        help='Add APR in epoch X, when adv_epoch is 0, it\'s equivalent to pure AMF.\n '
                             'And when adv_epoch is larger than epochs, it\'s equivalent to pure MF model. ')
    parser.add_argument('--adv', nargs='?', default='grad',
                        help='Generate the adversarial sample by gradient method or random method')
    parser.add_argument('--eps', type=float, default=0.5,
                        help='Epsilon for adversarial weights.')
    return parser.parse_args()


# data sampling and shuffling

# input: dataset(Mat, List, Rating, Negatives), batch_choice, num_negatives
# output: [_user_input_list, _item_input_pos_list]
def sampling(dataset):
    _user_input, _item_input_pos = [], []
    for (u, i) in dataset.trainMatrix.keys():
        # positive instance
        _user_input.append(u)
        _item_input_pos.append(i)
    return _user_input, _item_input_pos


def shuffle(samples, batch_size, dataset, model):
    global _user_input
    global _item_input_pos
    global _batch_size
    global _index
    global _model
    global _dataset
    _user_input, _item_input_pos = samples
    _batch_size = batch_size
    _index = range(len(_user_input))
    _model = model
    _dataset = dataset
    np.random.shuffle(_index)
    num_batch = len(_user_input) // _batch_size
    pool = Pool(cpu_count())
    res = pool.map(_get_train_batch, range(num_batch))
    pool.close()
    pool.join()
    user_list = [r[0] for r in res]
    item_pos_list = [r[1] for r in res]
    user_dns_list = [r[2] for r in res]
    item_dns_list = [r[3] for r in res]
    return user_list, item_pos_list, user_dns_list, item_dns_list


def _get_train_batch(i):
    user_batch, item_batch = [], []
    user_neg_batch, item_neg_batch = [], []
    begin = i * _batch_size
    for idx in range(begin, begin + _batch_size):
        user_batch.append(_user_input[_index[idx]])
        item_batch.append(_item_input_pos[_index[idx]])
        for dns in range(_model.dns):
            user = _user_input[_index[idx]]
            user_neg_batch.append(user)
            # negtive k
            gtItem = _dataset.testRatings[user][1]
            j = np.random.randint(_dataset.num_items)
            while j in _dataset.trainList[_user_input[_index[idx]]]:
                j = np.random.randint(_dataset.num_items)
            item_neg_batch.append(j)
    return np.array(user_batch)[:, None], np.array(item_batch)[:, None], \
           np.array(user_neg_batch)[:, None], np.array(item_neg_batch)[:, None]


# prediction model
class MF:
    def __init__(self, num_users, num_items, args):
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_size = args.embed_size
        self.learning_rate = args.lr
        self.reg = args.reg
        self.dns = args.dns
        self.adv = args.adv
        self.eps = args.eps
        self.adver = args.adver
        self.reg_adv = args.reg_adv
        self.epochs = args.epochs

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
            self.item_input_pos = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_pos")
            self.item_input_neg = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_neg")

    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.embedding_P = tf.Variable(
                tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                name='embedding_P', dtype=tf.float32)  # (users, embedding_size)
            self.embedding_Q = tf.Variable(
                tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                name='embedding_Q', dtype=tf.float32)  # (items, embedding_size)

            self.delta_P = tf.Variable(tf.zeros(shape=[self.num_users, self.embedding_size]),
                                       name='delta_P', dtype=tf.float32, trainable=False)  # (users, embedding_size)
            self.delta_Q = tf.Variable(tf.zeros(shape=[self.num_items, self.embedding_size]),
                                       name='delta_Q', dtype=tf.float32, trainable=False)  # (items, embedding_size)

            self.h = tf.constant(1.0, tf.float32, [self.embedding_size, 1], name="h")

    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            # embedding look up
            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input),
                                             1)  # (b, embedding_size)
            return tf.matmul(self.embedding_p * self.embedding_q, self.h), self.embedding_p, self.embedding_q # (b, embedding_size) * (embedding_size, 1)

    """ 1. create inference adv """

    def _create_inference_adv(self, item_input):
        with tf.name_scope("inference_adv"):
            # embedding look up
            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input),
                                             1)  # (b, embedding_size)
            # add adversarial noise
            self.P_plus_delta = self.embedding_p + tf.reduce_sum(tf.nn.embedding_lookup(self.delta_P, self.user_input),
                                                                 1)
            self.Q_plus_delta = self.embedding_q + tf.reduce_sum(tf.nn.embedding_lookup(self.delta_Q, item_input), 1)
            return tf.matmul(self.P_plus_delta * self.Q_plus_delta, self.h), self.embedding_p, self.embedding_q  # (b, embedding_size) * (embedding_size, 1)

    """ 2. loss 函数 """

    def _create_loss(self):
        with tf.name_scope("loss"):
            # loss for L(Theta)
            self.output, embed_p_pos, embed_q_pos = self._create_inference(self.item_input_pos)
            self.output_neg, embed_p_neg, embed_q_neg = self._create_inference(self.item_input_neg)
            self.result = tf.clip_by_value(self.output - self.output_neg, -80.0, 1e8)
            # self.loss = tf.reduce_sum(tf.log(1 + tf.exp(-self.result))) # this is numerically unstable
            self.loss = tf.reduce_sum(tf.nn.softplus(-self.result))

            # loss to be omptimized
            self.opt_loss = self.loss + self.reg * tf.reduce_mean(tf.square(embed_p_pos) + tf.square(embed_q_pos) + tf.square(embed_q_neg)) # embed_p_pos == embed_q_neg

            if self.adver:
                # loss for L(Theta + adv_Delta)
                self.output_adv, embed_p_pos, embed_q_pos = self._create_inference_adv(self.item_input_pos)
                self.output_neg_adv, embed_p_neg, embed_q_neg = self._create_inference_adv(self.item_input_neg)
                self.result_adv = tf.clip_by_value(self.output_adv - self.output_neg_adv, -80.0, 1e8)
                # self.loss_adv = tf.reduce_sum(tf.log(1 + tf.exp(-self.result_adv)))
                self.loss_adv = tf.reduce_sum(tf.nn.softplus(-self.result_adv))
                self.opt_loss += self.reg_adv * self.loss_adv + \
                                 self.reg * tf.reduce_mean(tf.square(embed_p_pos) + tf.square(embed_q_pos) + tf.square(embed_q_neg))

    """ 3. create adversarial """

    def _create_adversarial(self):
        with tf.name_scope("adversarial"):
            # generate the adversarial weights by random method
            if self.adv == "random":
                # generation
                self.adv_P = tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01)
                self.adv_Q = tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01)

                # normalization and multiply epsilon
                self.update_P = self.delta_P.assign(tf.nn.l2_normalize(self.adv_P, 1) * self.eps)
                self.update_Q = self.delta_Q.assign(tf.nn.l2_normalize(self.adv_Q, 1) * self.eps)

            # generate the adversarial weights by gradient-based method
            elif self.adv == "grad":
                # return the IndexedSlice Data: [(values, indices, dense_shape)]
                # grad_var_P: [grad,var], grad_var_Q: [grad, var]
                self.grad_P, self.grad_Q = tf.gradients(self.loss, [self.embedding_P, self.embedding_Q])

                # convert the IndexedSlice Data to Dense Tensor
                self.grad_P_dense = tf.stop_gradient(self.grad_P)
                self.grad_Q_dense = tf.stop_gradient(self.grad_Q)

                # normalization: new_grad = (grad / |grad|) * eps
                self.update_P = self.delta_P.assign(tf.nn.l2_normalize(self.grad_P_dense, 1) * self.eps)
                self.update_Q = self.delta_Q.assign(tf.nn.l2_normalize(self.grad_Q_dense, 1) * self.eps)

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.opt_loss)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()
        self._create_adversarial()
