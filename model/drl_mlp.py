# -*- coding: utf-8 -*-

import tensorflow as tf

class DRL:
    def __init__(self, batch_size, series_len, window, n_layer, n_hidden, gamma=1.0, transaction_cost=0.):
        self.batch_size = batch_size
        self.series_len = series_len
        self.window = window
        self.n_layer = n_layer
        self.n_hidden = n_hidden
        self.gamma = gamma

        self.price_series = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.series_len, self.window])
        ta_prices = tf.TensorArray(dtype=tf.float32, size=self.series_len)
        ta_prices = ta_prices.unstack(tf.transpose(self.price_series, perm=[1, 0, 2]))

        self.profit_series = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.series_len])
        ta_profits = tf.TensorArray(dtype=tf.float32, size=series_len)
        ta_profits = ta_profits.unstack(tf.transpose(self.profit_series, perm=[1, 0]))

        W = [tf.Variable(self.init_matrix(shape=[self.window, self.n_hidden]))]
        for _ in range(1, self.n_layer):
            W.append(tf.Variable(self.init_matrix(shape=[self.n_hidden, self.n_hidden])))
        W_out = tf.Variable(self.init_matrix(shape=[self.n_hidden, 1]))

        b = [tf.Variable(tf.constant(0., shape=[self.n_hidden]))]
        for _ in range(1, self.n_layer):
            b.append(tf.Variable(tf.constant(0., shape=[self.n_hidden])))
        b_out = tf.Variable(tf.constant(0., shape=[1]))

        self.ta_policy = tf.TensorArray(dtype=tf.float32, size=self.series_len)
        self.rewards = tf.TensorArray(dtype=tf.float32, size=self.series_len)

        output_prev = tf.zeros((self.batch_size, ))

        def _recurrence(i, out_prev, ta_policy, rewards):
            f = ta_prices.read(i)
            h = tf.matmul(f, W[0]) + b[0]
            for j in range(1, self.n_layer):
                h = tf.matmul(h, W[j]) + b[j]
            out = tf.nn.tanh(tf.matmul(h, W_out) + b_out)
            out = tf.reshape(out, shape=[self.batch_size])
            ta_policy = ta_policy.write(i, out)
            profit = ta_profits.read(i)
            reward = out * profit - transaction_cost * tf.abs(out - out_prev)
            rewards = rewards.write(i, reward)
            return i+1, out, ta_policy, rewards

        _, _, self.ta_policy, self.rewards = tf.while_loop(
            cond=lambda i, _1, _2, _3: i < self.series_len,
            body=_recurrence,
            loop_vars=(
                tf.constant(0, dtype=tf.int32), output_prev, self.ta_policy, self.rewards
            )
        )

        self.rewards = self.rewards.stack()
        self.ta_policy = self.ta_policy.stack()
        self.ta_policy = tf.transpose(self.ta_policy, perm=[1, 0])

        self.total_profit = tf.reduce_sum(self.rewards)
        self.train_op = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(-self.total_profit)

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

