from tensorflow import stack
import tensorflow as tf
from tensorflow.keras.utils import timeseries_dataset_from_array
import matplotlib.pyplot as plt
import numpy as np
from .train_functionalities import BATCH_SIZE

HISTORY_STEPS = 10
OUT_STEPS = 10


def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels


class WindowGenerator:
    def __init__(self, input_width, label_width, shift, train_df=None, val_df=None, test_df=None, label_columns=None,
                 val_percentage=None, test_percentage=None, batch_size=BATCH_SIZE, shuffle=False):

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        if not((val_percentage is not None and test_percentage is not None)
               or (train_df is not None and val_df is not None and test_df is not None)):
            raise Exception("Either give train_df, val_df, test_df or val_percentage, test_percentage")

        self.val_percentage = val_percentage
        self.test_percentage = test_percentage
        self.shuffle = shuffle

        if type(train_df) is dict:
            self.features = train_df[list(train_df.keys())[0]].columns
        else:
            self.features = train_df.columns

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(self.features)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.batch_size = batch_size
        self.total_batches = None
        self.train_batches = None
        self.val_batches = None
        self.test_batches = None

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


WindowGenerator.split_window = split_window


def make_dataset(self, data, shuffle):
    if type(data) is dict:
        ds_l = []
        for key in data.keys():
            dt_df = data[key].copy()
            dt = np.array(dt_df, dtype=np.float32)

            try:
                ds_ = timeseries_dataset_from_array(
                    data=dt,
                    targets=None,
                    sequence_length=self.total_window_size,
                    sequence_stride=1,
                    shuffle=shuffle,
                    batch_size=self.batch_size, )

                ds_ = ds_.map(self.split_window)
                ds_l.append(ds_)

            except ValueError:
                #print(f'Value Error: Failed to create dataset for dataframe split {key} with {len(dt_df)} elements '
                #      f'- Skipping.')
                pass

        ds = ds_l[0]
        ds_l.remove(ds_l[0])
        for ds_ in ds_l:
            ds = ds.concatenate(ds_)

        if self.total_batches is None:
            self.total_batches = len(ds)
            self.val_batches = max(1, int(np.floor(len(ds) * self.val_percentage)))
            self.test_batches = max(1, int(np.floor(len(ds) * self.test_percentage)))
            self.train_batches = self.total_batches - self.val_batches - self.test_batches
            #print(f'Train {self.train_batches} Val {self.val_batches} Test {self.test_batches}')

        # print(f'Total num batchs {len(ds)}')
        # print([x[0].shape for x in list(ds.as_numpy_iterator())])

    else:
        data = np.array(data, dtype=np.float32)
        ds = timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=self.batch_size, )
        ds = ds.map(self.split_window)

    return ds


WindowGenerator.make_dataset = make_dataset


@property
def complete(self):
    return self.make_dataset(self.train_df, shuffle=self.shuffle)


@property
def train(self):
    if self.val_percentage is not None or self.test_percentage is not None:
        return self.complete.take(self.train_batches)
    else:
        return self.make_dataset(self.train_df, self.shuffle)


@property
def val(self):
    if self.val_percentage is not None:
        return self.complete.skip(self.train_batches).take(self.val_batches)
    else:
        return self.make_dataset(self.val_df, self.shuffle)


@property
def test(self):
    if self.test_percentage is not None:
        return self.complete.skip(self.train_batches).skip(self.val_batches).take(self.test_batches)
    else:
        return self.make_dataset(self.test_df, self.shuffle)


@property
def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the `.test` dataset
        try:
            result = next(iter(self.test))
        except StopIteration:
            result = self.test.take(1)
            result = next(result.as_numpy_iterator())

        # And cache it for next time
        self._example = result
    return result


WindowGenerator.complete = complete
WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example


def plot(self, model=None, plot_col='Labels', max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(max_n, 1, n + 1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                 label='Input:' + plot_col, marker='.', zorder=-10)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue

        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

        if n == 0:
            plt.legend()

    plt.xlabel('Time [min]')
    plt.show()


WindowGenerator.plot = plot


def get_target_column_features(df, column_name):
    return df[column_name].values.reshape(-1, 1)


def multiple_point_window_generator(train_df, val_df, test_df, target_column, label_length, batch_size=BATCH_SIZE,
                                    val_percentage=None, test_percentage=None, shuffle=False):
    if test_percentage is not None and type(test_percentage) is not int:
        total_window_size = batch_size
    else:
        total_window_size = len(test_df)

    LABEL_WIDTH = total_window_size - label_length
    INPUT_WIDTH = total_window_size - label_length
    wide_window = WindowGenerator(input_width=INPUT_WIDTH, label_width=LABEL_WIDTH, shift=1, train_df=train_df,
                                  val_df=val_df, test_df=test_df, label_columns=[target_column],
                                  val_percentage=val_percentage, test_percentage=test_percentage, batch_size=batch_size,
                                  shuffle=shuffle)

    return wide_window


def single_step_window_generator(train_df, val_df, test_df, target_column, batch_size=BATCH_SIZE,
                                 val_percentage=None, test_percentage=None, shuffle=False):
    single_step_window = WindowGenerator(input_width=1, label_width=1, shift=1, train_df=train_df, val_df=val_df,
                                         test_df=test_df, label_columns=[target_column], val_percentage=val_percentage,
                                         test_percentage=test_percentage, batch_size=batch_size, shuffle=shuffle)
    return single_step_window


def conv_window_generator(train_df, val_df, test_df, target_column, history_length=HISTORY_STEPS,
                          batch_size=BATCH_SIZE, val_percentage=None, test_percentage=None, shuffle=False):
    conv_window_gen = WindowGenerator(input_width=history_length, label_width=1, shift=1, train_df=train_df,
                                      val_df=val_df, test_df=test_df, label_columns=[target_column],
                                      val_percentage=val_percentage, test_percentage=test_percentage, batch_size=batch_size,
                                      shuffle=shuffle)
    return conv_window_gen


def multiple_point_conv_window_generator(train_df, val_df, test_df, target_column, history_length=HISTORY_STEPS,
                                         batch_size=BATCH_SIZE, val_percentage=None, test_percentage=None, shuffle=False):
    if test_percentage is not None and type(test_percentage) is not int:
        total_window_size = batch_size
    else:
        total_window_size = len(test_df)

    LABEL_WIDTH = total_window_size - history_length
    INPUT_WIDTH = LABEL_WIDTH + (history_length - 1)
    wide_conv_window = WindowGenerator(input_width=INPUT_WIDTH, label_width=LABEL_WIDTH, shift=1, train_df=train_df,
                                       val_df=val_df, test_df=test_df, label_columns=[target_column],
                                       val_percentage=val_percentage, test_percentage=test_percentage, batch_size=batch_size,
                                       shuffle=shuffle)
    return wide_conv_window


def multi_step_window_generator(train_df, val_df, test_df, target_column, in_steps=HISTORY_STEPS, out_steps=OUT_STEPS,
                                batch_size=BATCH_SIZE,val_percentage=None, test_percentage=None, shuffle=False):
    multi_window = WindowGenerator(input_width=in_steps, label_width=out_steps, shift=out_steps, train_df=train_df,
                                   val_df=val_df, test_df=test_df, label_columns=[target_column],
                                   val_percentage=val_percentage, test_percentage=test_percentage, batch_size=batch_size,
                                   shuffle=shuffle)
    return multi_window
