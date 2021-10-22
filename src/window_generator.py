import tensorflow as tf

import pandas as pandas

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import data_division as dv


df = dv.load_data()
#df = dv.data_without_locations(db)
train_df, val_df, test_df, num_features = dv.db_division(df)
train_df, val_df, test_df, train_mean, train_std = dv.data_normalize(train_df, val_df, test_df)


column_indices = {name: i for i, name in enumerate(train_df.columns)}


#def columns_names():
#    return {name: i for i, name in enumerate(train_df.columns)}


def plot_db_normalized( df, train_mean, train_std ):
    df_std = (df - train_mean) / train_std
    df_std = df_std.melt(var_name = 'Column', value_name = 'Normalized')
    plt.figure(figsize = (12, 6))
    ax = sns.violinplot(x = 'Column', y = 'Normalized', data = df_std)
    _ = ax.set_xticklabels(df.keys(), rotation = 90)


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                train_df = train_df, val_df = val_df, test_df = test_df, label_columns = None):

        # armazena dados brutos.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # descobrir os índices da coluna do rótulo.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                                enumerate(train_df.columns)}

        # configurar parâmetros de janela.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]


    def __repr__( self ):
        return '\n'.join([
            f'Total tamanho de janela: {self.total_window_size}',
            f'Entrada indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label nome(s) das colunas: {self.label_columns}'])


    def split_window( self, features ):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis = -1)

        # Slicing não preserva a informação estática do formato, então defina manualmente. 
        # Use 'tf.data.Datasets' para facilitar.
        
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels


    def plot( self, model = None, plot_col = 'TEMPERATURA_DO_AR_BULBO_SECO_HORARIA', max_subplots = 3 ):
        inputs, labels = self.example
        plt.figure(figsize = (12, 8))
        plot_col_index = self.column_indices[plot_col]

        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                        label = 'Inputs', marker = '.', zorder = -10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors = 'k', label = 'Labels', c = '#2ca02c', s = 64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker = 'X', edgecolors='k', label = 'Predictions',
                            c = '#ff7f0e', s = 64)

            if n == 0:
                plt.legend()

            plt.xlabel('Time [h]')

        plt.show()


    def make_dataset( self, data ):
        data = np.array(data, dtype = np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data = data,
                targets = None,
                sequence_length = self.total_window_size,
                sequence_stride = 1,
                shuffle = True,
                batch_size = 64,)

        ds = ds.map(self.split_window)

        return ds

    
    @property
    def train( self ):
        return self.make_dataset(self.train_df)


    @property
    def val( self ):
        return self.make_dataset(self.val_df)


    @property
    def test( self ):
        return self.make_dataset(self.test_df)

    
    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result
