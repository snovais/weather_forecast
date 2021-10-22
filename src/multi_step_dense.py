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

# Configure um objeto WindowGenerator para produzir pares de etapa única (input, label)
single_step_window = wg(
    input_width = 1, label_width = 1, shift = 1,
    label_columns = [col])


def compile_and_fit( model, window, patience = 2 ):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                                    patience = patience,
                                                    mode = 'min')

    model.compile(loss = tf.losses.MeanSquaredError(),
                optimizer = tf.optimizers.Adamax(),
                metrics = [tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs = MAX_EPOCHS,
                        validation_data = window.val,
                        callbacks = [early_stopping])
    return history


val_performance = {}
performance = {}

wide_window = wg(
    input_width=24, label_width=24, shift=1,
    label_columns=[col])

    
CONV_WIDTH = 3
conv_window = wg(
    input_width = CONV_WIDTH,
    label_width = 1,
    shift = 1,
    label_columns = [col])


conv_window.plot()
plt.title("Given 3h as input, predict 1h into the future.")
plt.show()

MAX_EPOCHS = 200

multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units = 32, activation = 'relu'),
    tf.keras.layers.Dense(units = 32, activation = 'relu'),
    tf.keras.layers.Dense(units = 1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])


#print('Input shape:', conv_window.example[0].shape)
#print('Output shape:', multi_step_dense(conv_window.example[0]).shape)

print('Input shape:', wide_window.example[0].shape)
try:
  print('Output shape:', multi_step_dense(wide_window.example[0]).shape)
except Exception as e:
  print(f'\n{type(e).__name__}:{e}')

history = compile_and_fit(multi_step_dense, conv_window)

val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose = 0)


conv_window.plot(multi_step_dense)
