# -*- coding: utf-8 -*-

import tensorflow as tf

def create_hparams():
    hparams = tf.contrib.training.HParams(
        series_len=12,
        window=12,
        n_hidden=32,
    )
    return hparams