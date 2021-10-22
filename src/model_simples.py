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

#print(single_step_window)

#for example_inputs, example_labels in single_step_window.train.take(1):
#    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
#    print(f'Labels shape (batch, time, features): {example_labels.shape}')


class Baseline( tf.keras.Model ):
    def __init__( self, label_index = None ):
        super().__init__()
        self.label_index = label_index

    def call( self, inputs ):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


baseline = Baseline(label_index = column_indices[col])

baseline.compile(loss = tf.losses.MeanSquaredError(),
                 metrics = [tf.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose = 0)

wide_window = wg(
    input_width=24, label_width=24, shift=1,
    label_columns=[col])

print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)

#wide_window.plot(baseline)

linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units = 1),
    tf.keras.layers.Dense(units = 10)
])

print('Input shape:', single_step_window.example[0].shape)
print('Output shape:', linear(single_step_window.example[0]).shape)

MAX_EPOCHS = 100

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


history = compile_and_fit(linear, single_step_window)

val_performance['Linear'] = linear.evaluate(single_step_window.val)
performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)

print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)

wide_window.plot(linear)

plt.bar(x = range(len(train_df.columns)),
        height = linear.layers[0].kernel[:, 0].numpy())
axis = plt.gca()
axis.set_xticks(range(len(train_df.columns)))
_ = axis.set_xticklabels(train_df.columns, rotation = 90)
plt.show()
