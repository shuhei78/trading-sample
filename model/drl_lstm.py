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
        self.transaction_cost = transaction_cost

        self.params = []
        self.recurrent_unit = self.create_recurrent_unit(self.params)
        self.output_unit = self.create_output_unit(self.params)

        self.price_series = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.series_len, self.window])
        ta_prices = tf.TensorArray(dtype=tf.float32, size=self.series_len)
        ta_prices = ta_prices.unstack(tf.transpose(self.price_series, perm=[1, 0, 2]))

        self.profit_series = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.series_len])
        ta_profits = tf.TensorArray(dtype=tf.float32, size=series_len)
        ta_profits = ta_profits.unstack(tf.transpose(self.profit_series, perm=[1, 0]))

        self.ta_policy = tf.TensorArray(dtype=tf.float32, size=self.series_len)
        self.rewards = tf.TensorArray(dtype=tf.float32, size=self.series_len)

        output_prev = tf.zeros((self.batch_size, ))

        h0 = tf.zeros([self.batch_size, self.n_hidden])
        self.h0 = tf.stack([h0, h0])

        # This is test
        # self.dropout_keep_prob = tf.placeholder(dtype=tf.float32)

        def _recurrence(i, h_prev, out_prev, ta_policy, rewards):
            f = ta_prices.read(i)
            h = self.recurrent_unit(f, h_prev)
            out = self.output_unit(h)
            out = tf.reshape(out, shape=[self.batch_size])
            ta_policy = ta_policy.write(i, out)
            profit = ta_profits.read(i)
            reward = out * profit - self.transaction_cost * tf.abs(out - out_prev)
            rewards = rewards.write(i, reward)
            return i+1, h, out, ta_policy, rewards

        _, _, _, self.ta_policy, self.rewards = tf.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.series_len,
            body=_recurrence,
            loop_vars=(
                tf.constant(0, dtype=tf.int32), self.h0, output_prev, self.ta_policy, self.rewards
            )
        )

        self.rewards = self.rewards.stack()
        self.ta_policy = self.ta_policy.stack()
        self.ta_policy = tf.transpose(self.ta_policy, perm=[1, 0])

        self.total_profit = tf.reduce_sum(self.rewards)
        self.train_op = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(-self.total_profit)
        # self.train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(-self.total_profit)

        # optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        # grad, _ = tf.clip_by_global_norm(tf.gradients(-self.total_profit, self.params), clip_norm=5.0)
        # self.train_op = optimizer.apply_gradients(zip(grad, self.params))


    def init_matrix(self, shape):
        # return tf.truncated_normal(shape, stddev=0.1)
        return tf.random_normal(shape, stddev=0.1)

    def create_recurrent_unit(self, params):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.Variable(self.init_matrix([self.window, self.n_hidden]))
        self.Ui = tf.Variable(self.init_matrix([self.n_hidden, self.n_hidden]))
        self.bi = tf.Variable(self.init_matrix([self.n_hidden]))

        self.Wf = tf.Variable(self.init_matrix([self.window, self.n_hidden]))
        self.Uf = tf.Variable(self.init_matrix([self.n_hidden, self.n_hidden]))
        self.bf = tf.Variable(self.init_matrix([self.n_hidden]))

        self.Wog = tf.Variable(self.init_matrix([self.window, self.n_hidden]))
        self.Uog = tf.Variable(self.init_matrix([self.n_hidden, self.n_hidden]))
        self.bog = tf.Variable(self.init_matrix([self.n_hidden]))

        self.Wc = tf.Variable(self.init_matrix([self.window, self.n_hidden]))
        self.Uc = tf.Variable(self.init_matrix([self.n_hidden, self.n_hidden]))
        self.bc = tf.Variable(self.init_matrix([self.n_hidden]))
        params.extend([
            self.Wi, self.Ui, self.bi,
            self.Wf, self.Uf, self.bf,
            self.Wog, self.Uog, self.bog,
            self.Wc, self.Uc, self.bc])

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self, params):
        self.Wo = tf.Variable(self.init_matrix([self.n_hidden, 1]))
        self.bo = tf.Variable(self.init_matrix([1]))
        params.extend([self.Wo, self.bo])

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch_size * n_hidden
            # hidden_state = tf.nn.dropout(hidden_state, keep_prob=self.dropout_keep_prob)
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            output = tf.nn.tanh(logits)
            return output

        return unit
