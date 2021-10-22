# -*- coding: utf-8 -*-

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt
from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')

from window_generator import train_df, column_indices, WindowGenerator as wg


col = 'TEMPERATURA_DO_AR_BULBO_SECO_HORARIA'


def compile_and_fit( model, window, patience = 2 ):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                    patience = patience,
                                                    )#mode = 'min')

    model.compile(loss = tf.losses.MeanSquaredError(),
                optimizer = tf.optimizers.Adamax(learning_rate = 0.001),
                metrics = [tf.metrics.RootMeanSquaredError()])

    history = model.fit(window.train, epochs = MAX_EPOCHS,
                        validation_data = window.val)#,
                        #callbacks = [early_stopping])
    return history


MAX_EPOCHS = 30


val_performance = {}
performance = {}

wide_window = wg(
    input_width = 24, label_width = 24, shift = 1,
    label_columns = [col])


neurons = 1024
lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(64, dropout = 0.1, recurrent_dropout = 0.1, return_sequences = True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units = neurons, activation = 'relu'),
    tf.keras.layers.Dense(units = neurons, activation = 'relu'),
    tf.keras.layers.Dense(units = 1)
])


history = compile_and_fit(lstm_model, wide_window)

val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose = 0)

print(performance['LSTM'])

wide_window.plot(lstm_model)
