from __future__ import absolute_import
from __future__ import division
import os
import math
import numpy as np
import tensorflow as tf
from multiprocessing import Pool
from multiprocessing import cpu_count
import argparse
import logging
from time import time
from time import strftime
from time import localtime
from Dataset import Dataset

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

# training
def training(model, dataset, args, epoch_start, epoch_end, time_stamp):  # saver is an object to save pq
    with tf.Session() as sess:
        # initialized the save op
        if args.adver:
            ckpt_save_path = "Pretrain/%s/APR/embed_%d/%s/" % (args.dataset, args.embed_size, time_stamp)
            ckpt_restore_path = "Pretrain/%s/MF_BPR/embed_%d/%s/" % (args.dataset, args.embed_size, time_stamp)
        else:
            ckpt_save_path = "Pretrain/%s/MF_BPR/embed_%d/%s/" % (args.dataset, args.embed_size, time_stamp)
            ckpt_restore_path = 0 if args.restore is None else "Pretrain/%s/MF_BPR/embed_%d/%s/" % (args.dataset, args.embed_size, args.restore)

        if not os.path.exists(ckpt_save_path):
            os.makedirs(ckpt_save_path)
        if ckpt_restore_path and not os.path.exists(ckpt_restore_path):
            os.makedirs(ckpt_restore_path)

        saver_ckpt = tf.train.Saver({'embedding_P': model.embedding_P, 'embedding_Q': model.embedding_Q})

        # pretrain or not
        sess.run(tf.global_variables_initializer())

        # restore the weights when pretrained
        if args.restore is not None or epoch_start:
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_restore_path + 'checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver_ckpt.restore(sess, ckpt.model_checkpoint_path)
        # initialize the weights
        else:
            logging.info("Initialized from scratch")
            print ("Initialized from scratch")

        # initialize for Evaluate
        eval_feed_dicts = init_eval_model(model, dataset)

        # sample the data
        samples = sampling(dataset)

        # initialize the max_ndcg to memorize the best result
        max_ndcg = 0
        best_res = {}

        # train by epoch
        for epoch_count in range(epoch_start, epoch_end+1):

            # initialize for training batches
            batch_begin = time()
            batches = shuffle(samples, args.batch_size, dataset, model)
            batch_time = time() - batch_begin

            # compute the accuracy before training
            prev_batch = batches[0], batches[1], batches[3]
            _, prev_acc = training_loss_acc(model, sess, prev_batch, output_adv=0)

            # training the model
            train_begin = time()
            train_batches = training_batch(model, sess, batches, args.adver)
            train_time = time() - train_begin

            if epoch_count % args.verbose == 0:
                _, ndcg, cur_res = output_evaluate(model, sess, dataset, train_batches, eval_feed_dicts,
                                                   epoch_count, batch_time, train_time, prev_acc, output_adv=0)

            # print and log the best result
            if max_ndcg < ndcg:
                max_ndcg = ndcg
                best_res['result'] = cur_res
                best_res['epoch'] = epoch_count

            if model.epochs == epoch_count:
                print ("Epoch %d is the best epoch" % best_res['epoch'])
                for idx, (hr_k, ndcg_k, auc_k) in enumerate(np.swapaxes(best_res['result'], 0, 1)):
                    res = "K = %d: HR = %.4f, NDCG = %.4f AUC = %.4f" % (idx + 1, hr_k, ndcg_k, auc_k)
                    print (res)

            # save the embedding weights
            if args.ckpt > 0 and epoch_count % args.ckpt == 0:
                saver_ckpt.save(sess, ckpt_save_path + 'weights', global_step=epoch_count)

        saver_ckpt.save(sess, ckpt_save_path + 'weights', global_step=epoch_count)


def output_evaluate(model, sess, dataset, train_batches, eval_feed_dicts, epoch_count, batch_time, train_time, prev_acc,
                    output_adv):
    loss_begin = time()
    train_loss, post_acc = training_loss_acc(model, sess, train_batches, output_adv)
    loss_time = time() - loss_begin

    eval_begin = time()
    result = evaluate(model, sess, dataset, eval_feed_dicts, output_adv)
    eval_time = time() - eval_begin

    # check embedding
    embedding_P, embedding_Q = sess.run([model.embedding_P, model.embedding_Q])

    hr, ndcg, auc = np.swapaxes(result, 0, 1)[-1]
    res = "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f ACC = %.4f ACC_adv = %.4f [%.1fs], |P|=%.2f, |Q|=%.2f" % \
          (epoch_count, batch_time, train_time, hr, ndcg, prev_acc,
           post_acc, eval_time, np.linalg.norm(embedding_P), np.linalg.norm(embedding_Q))

    print (res)

    return post_acc, ndcg, result


# input: batch_index (shuffled), model, sess, batches
# do: train the model optimizer
def training_batch(model, sess, batches, adver=False):
    user_input, item_input_pos, user_dns_list, item_dns_list = batches
    # dns for every mini-batch
    # dns = 1, i.e., BPR
    if model.dns == 1:
        item_input_neg = item_dns_list
        # for BPR training
        for i in range(len(user_input)):
            feed_dict = {model.user_input: user_input[i],
                         model.item_input_pos: item_input_pos[i],
                         model.item_input_neg: item_input_neg[i]}
            if adver:
                sess.run([model.update_P, model.update_Q], feed_dict)
            sess.run(model.optimizer, feed_dict)
    # dns > 1, i.e., BPR-dns
    elif model.dns > 1:
        item_input_neg = []
        for i in range(len(user_input)):
            # get the output of negtive sample
            feed_dict = {model.user_input: user_dns_list[i],
                         model.item_input_neg: item_dns_list[i]}
            output_neg = sess.run(model.output_neg, feed_dict)
            # select the best negtive sample as for item_input_neg
            item_neg_batch = []
            for j in range(0, len(output_neg), model.dns):
                item_index = np.argmax(output_neg[j: j + model.dns])
                item_neg_batch.append(item_dns_list[i][j: j + model.dns][item_index][0])
            item_neg_batch = np.array(item_neg_batch)[:, None]
            # for mini-batch BPR training
            feed_dict = {model.user_input: user_input[i],
                         model.item_input_pos: item_input_pos[i],
                         model.item_input_neg: item_neg_batch}
            sess.run(model.optimizer, feed_dict)
            item_input_neg.append(item_neg_batch)
    return user_input, item_input_pos, item_input_neg


# calculate the gradients
# update the adversarial noise
def adv_update(model, sess, train_batches):
    user_input, item_input_pos, item_input_neg = train_batches
    # reshape mini-batches into a whole large batch
    user_input, item_input_pos, item_input_neg = \
        np.reshape(user_input, (-1, 1)), np.reshape(item_input_pos, (-1, 1)), np.reshape(item_input_neg, (-1, 1))
    feed_dict = {model.user_input: user_input,
                 model.item_input_pos: item_input_pos,
                 model.item_input_neg: item_input_neg}

    return sess.run([model.update_P, model.update_Q], feed_dict)


# input: model, sess, batches
# output: training_loss
def training_loss_acc(model, sess, train_batches, output_adv):
    train_loss = 0.0
    acc = 0
    num_batch = len(train_batches[1])
    user_input, item_input_pos, item_input_neg = train_batches
    for i in range(len(user_input)):
        # print user_input[i][0]. item_input_pos[i][0], item_input_neg[i][0]
        feed_dict = {model.user_input: user_input[i],
                     model.item_input_pos: item_input_pos[i],
                     model.item_input_neg: item_input_neg[i]}
        if output_adv:
            loss, output_pos, output_neg = sess.run([model.loss_adv, model.output_adv, model.output_neg_adv], feed_dict)
        else:
            loss, output_pos, output_neg = sess.run([model.loss, model.output, model.output_neg], feed_dict)
        train_loss += loss
        acc += ((output_pos - output_neg) > 0).sum() / len(output_pos)
    return train_loss / num_batch, acc / num_batch


def init_eval_model(model, dataset):
    begin_time = time()
    global _dataset
    global _model
    _dataset = dataset
    _model = model

    pool = Pool(cpu_count())
    feed_dicts = pool.map(_evaluate_input, range(_dataset.num_users))
    pool.close()
    pool.join()

    print("Load the evaluation model done [%.1f s]" % (time() - begin_time))
    return feed_dicts


def _evaluate_input(user):
    # generate items_list
    test_item = _dataset.testRatings[user][1]
    item_input = set(range(_dataset.num_items)) - set(_dataset.trainList[user])
    if test_item in item_input:
        item_input.remove(test_item)
    item_input = list(item_input)
    item_input.append(test_item)
    user_input = np.full(len(item_input), user, dtype='int32')[:, None]
    item_input = np.array(item_input)[:, None]
    return user_input, item_input


def evaluate(model, sess, dataset, feed_dicts, output_adv):
    global _model
    global _K
    global _sess
    global _dataset
    global _feed_dicts
    global _output
    _dataset = dataset
    _model = model
    _sess = sess
    _K = 100
    _feed_dicts = feed_dicts
    _output = output_adv

    res = []
    for user in range(_dataset.num_users):
        res.append(_eval_by_user(user))
    res = np.array(res)
    hr, ndcg, auc = (res.mean(axis=0)).tolist()

    return hr, ndcg, auc


def _eval_by_user(user):
    # get prredictions of data in testing set
    user_input, item_input = _feed_dicts[user]
    feed_dict = {_model.user_input: user_input, _model.item_input_pos: item_input}
    if _output:
        predictions = _sess.run(_model.output_adv, feed_dict)
    else:
        predictions = _sess.run(_model.output, feed_dict)

    neg_predict, pos_predict = predictions[:-1], predictions[-1]
    position = (neg_predict >= pos_predict).sum()

    # calculate from HR@1 to HR@100, and from NDCG@1 to NDCG@100, AUC
    hr, ndcg, auc = [], [], []
    for k in range(1, _K + 1):
        hr.append(position < k)
        ndcg.append(math.log(2) / math.log(position + 2) if position < k else 0)
        auc.append(1 - (position / len(neg_predict)))  # formula: [#(Xui>Xuj) / #(Items)] = [1 - #(Xui<=Xuj) / #(Items)]

    return hr, ndcg, auc

def init_logging(args, time_stamp):
    path = "Log/%s_%s/" % (strftime('%Y-%m-%d_%H', localtime()), args.task)
    if not os.path.exists(path):
        os.makedirs(path)
    logging.basicConfig(filename=path + "%s_log_embed_size%d_%s" % (args.dataset, args.embed_size, time_stamp),
                        level=logging.INFO)
    logging.info(args)
    print (args)


if __name__ == '__main__':

    time_stamp = strftime('%Y_%m_%d_%H_%M_%S', localtime())

    # initilize arguments and logging
    args = parse_args()
    init_logging(args, time_stamp)

    # initialize dataset
    dataset = Dataset(args.path + args.dataset)

    args.adver = 0
    # initialize MF_BPR models
    MF_BPR = MF(dataset.num_users, dataset.num_items, args)
    MF_BPR.build_graph()

    print ("Initialize MF_BPR")

    # start training
    training(MF_BPR, dataset, args, epoch_start=0, epoch_end=args.adv_epoch-1, time_stamp=time_stamp)

    args.adver = 1
    # instialize AMF model
    AMF = MF(dataset.num_users, dataset.num_items, args)
    AMF.build_graph()

    print ("Initialize AMF")

    # start training
    training(AMF, dataset, args, epoch_start=args.adv_epoch, epoch_end=args.epochs, time_stamp=time_stamp)
