# -*- coding: utf-8 -*-

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt
from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')

from window_generator import train_df, column_indices, example_window, example_inputs, example_labels, WindowGenerator as wg


col = 'TEMPERATURA_DO_AR_BULBO_SECO_HORARIA'


def compile_and_fit( model, window, patience = 2 ):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                    patience = patience,
                                                    )#mode = 'min')

    model.compile(loss = tf.losses.MeanSquaredError(),
                optimizer = tf.optimizers.Adamax(learning_rate = 0.01),
                metrics = [tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs = MAX_EPOCHS,
                        validation_data = window.val)#,
                        #callbacks = [early_stopping])
    return history


MAX_EPOCHS = 200


val_performance = {}
performance = {}

wide_window = wg(
    input_width = 2400, label_width = 2400, shift = 1,
    label_columns = [col])

tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True) # Enable XLA.


neurons = 256
gru_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, gru_units]
    tf.keras.layers.GRU(128, dropout = 0.3, recurrent_dropout = 0.3, return_sequences = True),
    tf.keras.layers.GRU(128, dropout = 0.3, recurrent_dropout = 0.3, return_sequences = True),
    tf.keras.layers.GRU(128, dropout = 0.3, recurrent_dropout = 0.3, return_sequences = True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units = neurons*2, activation = 'relu'),
    tf.keras.layers.Dense(units = 1)
])


print('Input shape:', wide_window.example[0].shape)
print('Output shape:', gru_model(wide_window.example[0]).shape)


gru_model.summary()

history = compile_and_fit(gru_model, wide_window)

val_performance['GRU'] = gru_model.evaluate(wide_window.val)
performance['GRU'] = gru_model.evaluate(wide_window.test, verbose = 0)

print(performance['GRU'])


wide_window.plot(gru_model)
