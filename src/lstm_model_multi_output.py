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


MAX_EPOCHS = 250


val_performance = {}
performance = {}

OUT_STEPS = 120
multi_window = wg(input_width = 120,
                               label_width = OUT_STEPS,
                               shift = OUT_STEPS)

#multi_window.plot()


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


def visualize_loss( history, title ):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


num_features = 4


multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(1024, return_sequences = False, dropout = 0.3, recurrent_dropout = 0.3,),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer = tf.initializers.zeros()),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer = tf.initializers.zeros()),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer = tf.initializers.zeros()),
    tf.keras.layers.Dropout(0.3),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_lstm_model, multi_window)


val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_lstm_model)

visualize_loss(history, "Training and Validation Loss")

