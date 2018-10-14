# -*- coding: utf-8 -*-

import tensorflow as tf

class DRL:
    def __init__(self, series_len, window, n_hidden, gamma=1.0, transaction_cost=0.):
        self.series_len = series_len
        self.window = window
        self.n_hidden = n_hidden
        self.gamma = gamma
        self.transaction_cost = transaction_cost

        self.params_1 = []
        self.params_2 = []
        self.params_out = []
        self.create_recurrent_unit(self.window, self.n_hidden, self.params_1)
        self.create_recurrent_unit(self.n_hidden, self.n_hidden, self.params_2)
        self.create_output_unit(self.params_out)

        self.price_series = tf.placeholder(dtype=tf.float32, shape=[None, self.series_len, self.window])
        ta_prices = tf.TensorArray(dtype=tf.float32, size=self.series_len)
        ta_prices = ta_prices.unstack(tf.transpose(self.price_series, perm=[1, 0, 2]))

        self.profit_series = tf.placeholder(dtype=tf.float32, shape=[None, self.series_len])
        ta_profits = tf.TensorArray(dtype=tf.float32, size=series_len)
        ta_profits = ta_profits.unstack(tf.transpose(self.profit_series, perm=[1, 0]))

        self.ta_policy = tf.TensorArray(dtype=tf.float32, size=self.series_len)
        self.rewards = tf.TensorArray(dtype=tf.float32, size=self.series_len)

        output_zeros_dims = tf.stack([tf.shape(self.price_series)[0], ])
        output_prev = tf.fill(output_zeros_dims, 0.0)

        zeros_dims = tf.stack([tf.shape(self.price_series)[0], self.n_hidden])
        h0 = tf.fill(zeros_dims, 0.0)
        self.h0 = tf.stack([h0, h0])

        h1 = tf.fill(zeros_dims, 0.0)
        self.h1 = tf.stack([h1, h1])

        # self.dropout_keep_prob = tf.placeholder(dtype=tf.float32)

        def _recurrence(i, h0_prev, h1_prev, out_prev, ta_policy, rewards):
            f = ta_prices.read(i)
            h0 = self.recurrent_unit(f, h0_prev, self.params_1)
            h1 = self.recurrent_unit(tf.unstack(h0)[0], h1_prev, self.params_2)
            out = self.output_unit(h1, self.params_out)
            out = tf.reshape(out, shape=[-1])
            ta_policy = ta_policy.write(i, out)
            profit = ta_profits.read(i)
            reward = out * profit - self.transaction_cost * tf.abs(out - out_prev)
            rewards = rewards.write(i, reward)
            return i+1, h0, h1, out, ta_policy, rewards

        _, _, _, _, self.ta_policy, self.rewards = tf.while_loop(
            cond=lambda i, _1, _2, _3, _4, _5: i < self.series_len,
            body=_recurrence,
            loop_vars=(
                tf.constant(0, dtype=tf.int32), self.h0, self.h1, output_prev, self.ta_policy, self.rewards
            )
        )

        self.rewards = self.rewards.stack()

        self.ta_policy = self.ta_policy.stack()
        self.ta_policy = tf.transpose(self.ta_policy, perm=[1, 0])

        self.total_profit = tf.reduce_sum(self.rewards)
        # self.train_op = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(-self.total_profit)
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(-self.total_profit)

    def init_matrix(self, shape):
        # return tf.truncated_normal(shape, stddev=0.1)
        return tf.random_normal(shape, stddev=0.1)

    def create_recurrent_unit(self, in_dim, out_dim, params):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.Variable(self.init_matrix([in_dim, out_dim]))
        self.Ui = tf.Variable(self.init_matrix([out_dim, out_dim]))
        self.bi = tf.Variable(self.init_matrix([out_dim]))

        self.Wf = tf.Variable(self.init_matrix([in_dim, out_dim]))
        self.Uf = tf.Variable(self.init_matrix([out_dim, out_dim]))
        self.bf = tf.Variable(self.init_matrix([out_dim]))

        self.Wog = tf.Variable(self.init_matrix([in_dim, out_dim]))
        self.Uog = tf.Variable(self.init_matrix([out_dim, out_dim]))
        self.bog = tf.Variable(self.init_matrix([out_dim]))

        self.Wc = tf.Variable(self.init_matrix([in_dim, out_dim]))
        self.Uc = tf.Variable(self.init_matrix([out_dim, out_dim]))
        self.bc = tf.Variable(self.init_matrix([out_dim]))
        params.extend([
            self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wog, self.Uog, self.bog,
            self.Wc, self.Uc, self.bc])

    def recurrent_unit(self, x, hidden_memory_tm1, params):
        previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

        # Input Gate
        i = tf.sigmoid(
            tf.matmul(x, params[0]) +
            tf.matmul(previous_hidden_state, params[1]) + params[2]
        )

        # Forget Gate
        f = tf.sigmoid(
            tf.matmul(x, params[3]) +
            tf.matmul(previous_hidden_state, params[4]) + params[5]
        )

        # Output Gate
        o = tf.sigmoid(
            tf.matmul(x, params[6]) +
            tf.matmul(previous_hidden_state, params[7]) + params[8]
        )

        # New Memory Cell
        c_ = tf.nn.tanh(
            tf.matmul(x, params[9]) +
            tf.matmul(previous_hidden_state, params[10]) + params[11]
        )

        # Final Memory cell
        c = f * c_prev + i * c_

        # Current Hidden state
        current_hidden_state = o * tf.nn.tanh(c)

        return tf.stack([current_hidden_state, c])

    def create_output_unit(self, params):
        self.Wo = tf.Variable(self.init_matrix([self.n_hidden, 1]))
        self.bo = tf.Variable(self.init_matrix([1]))
        params.extend([self.Wo, self.bo])

    def output_unit(self, hidden_memory_tuple, params):
        hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
        # hidden_state : batch_size * n_hidden
        # hidden_state = tf.nn.dropout(hidden_state, keep_prob=self.dropout_keep_prob)
        logits = tf.matmul(hidden_state, params[0]) + params[1]
        output = tf.nn.tanh(logits)
        return output

