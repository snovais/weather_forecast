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
                optimizer = tf.optimizers.Adamax(),
                metrics = [tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs = MAX_EPOCHS,
                        validation_data = window.val)#,
                        #callbacks = [early_stopping])
    return history


MAX_EPOCHS = 50


val_performance = {}
performance = {}

wide_window = wg(
    input_width = 24, label_width = 24, shift = 1,
    label_columns = [col])

    
CONV_WIDTH = 3
conv_window = wg(
    input_width = CONV_WIDTH,
    label_width = 1,
    shift = 1,
    label_columns = [col])


LABEL_WIDTH = 48
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = wg(
    input_width = INPUT_WIDTH,
    label_width = LABEL_WIDTH,
    shift = 1,
    label_columns = [col])

wide_conv_window


neurons = 1024
drop_out = 0.3
conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters = neurons,
                           kernel_size = (CONV_WIDTH,),
                           activation = 'relu'),
    tf.keras.layers.Dense(units = neurons, activation = 'relu'),
    tf.keras.layers.Dropout(drop_out),
    tf.keras.layers.Dense(units = neurons, activation = 'relu'),
    tf.keras.layers.Dropout(drop_out),
    tf.keras.layers.Dense(units = neurons, activation = 'relu'),
    tf.keras.layers.Dropout(drop_out),
    tf.keras.layers.Dense(units = neurons, activation = 'relu'),
    tf.keras.layers.Dropout(drop_out),
    tf.keras.layers.Dense(units = neurons, activation = 'relu'),
    tf.keras.layers.Dropout(drop_out),
    tf.keras.layers.Dense(units = neurons, activation = 'relu'),
    tf.keras.layers.Dropout(drop_out),
    tf.keras.layers.Dense(units = 1)
])


print("Wide conv window")
print('Input shape:', wide_conv_window.example[0].shape)
print('Labels shape:', wide_conv_window.example[1].shape)
print('Output shape:', conv_model(wide_conv_window.example[0]).shape)


history = compile_and_fit(conv_model, conv_window)

val_performance['Conv'] = conv_model.evaluate(conv_window.val)
performance['Conv'] = conv_model.evaluate(conv_window.test, verbose = 0, workers = 1, use_multiprocessing = True)

print(performance['Conv'])

wide_conv_window.plot(conv_model)
