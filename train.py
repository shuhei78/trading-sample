# -*- coding: utf-8 -*-

import tensorflow as tf
from sklearn.model_selection import train_test_split

from model.drl_stack_lstm import DRL
from data_loader import ChartLoader
from hparams import create_hparams

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('epoch_num', 1000, '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_string('result_dir', './result/', '')

hparams = create_hparams()

chart_loader = ChartLoader(n_window=hparams.window, seq_len=hparams.series_len)
train_feature, test_feature, train_profit, test_profit, train_price, test_price = chart_loader.get_samples()

train_feature, valid_feature, train_profit, valid_profit, train_price, valid_price\
    = train_test_split(train_feature, train_profit, train_price, test_size=0.1, random_state=0)

print(train_feature.shape, train_profit.shape)
print(valid_feature.shape, valid_profit.shape)
print(test_feature.shape, test_profit.shape)

train_batch = len(train_feature) // FLAGS.batch_size
valid_batch = len(valid_feature) // FLAGS.batch_size
test_batch = len(test_feature) // FLAGS.batch_size

drl = DRL(
    series_len=hparams.series_len,
    window=hparams.window,
    n_hidden=hparams.n_hidden
)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

def get_rewards(sess, features, profits, batch_num):
    rewards = []
    for i in range(batch_num):
        features_batch = features[i*FLAGS.batch_size: (i+1)*FLAGS.batch_size]
        profits_batch = profits[i*FLAGS.batch_size: (i+1)*FLAGS.batch_size]
        feed_dict = {
            drl.price_series: features_batch,
            drl.profit_series: profits_batch
        }
        reward = sess.run(drl.total_profit, feed_dict=feed_dict)
        rewards.append(reward)
    rewards_avg = sum(rewards) / len(rewards)
    rewards_avg = rewards_avg * (chart_loader.max_ - chart_loader.min_) / 2.

    return rewards_avg

for epoch in range(FLAGS.epoch_num):
    for i in range(train_batch):
        features_batch = train_feature[i*FLAGS.batch_size: (i+1)*FLAGS.batch_size]
        profits_batch = train_profit[i*FLAGS.batch_size: (i+1)*FLAGS.batch_size]
        feed_dict = {
            drl.price_series: features_batch,
            drl.profit_series: profits_batch,
            # drl.dropout_keep_prob: 1.0
        }
        _ = sess.run(drl.train_op, feed_dict=feed_dict)

    if epoch % 10 == 0:
        train_rewards = get_rewards(sess, train_feature, train_profit, train_batch)
        valid_rewards = get_rewards(sess, valid_feature, valid_profit, valid_batch)
        test_rewards = get_rewards(sess, test_feature, test_profit, test_batch)
        print("EPOCH: %s, TRAIN: %s, VALID: %s, TEST: %s" % (epoch, train_rewards, valid_rewards, test_rewards))

        # model_save_dir = FLAGS.result_dir + "model-%s/model.ckpt" % epoch
        # saver.save(sess, model_save_dir)
