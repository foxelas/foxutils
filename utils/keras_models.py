import tensorflow as tf
from tensorflow.keras import Model as kerasModel
from tensorflow.keras.callbacks import EarlyStopping as tfEarlyStopping
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam

from .utils import SEED
from .utils_data_generators import HISTORY_STEPS, BATCH_SIZE, OUT_STEPS
from .utils_train import MAX_EPOCHS

import matplotlib.pyplot as plt

set_random_seed(SEED)

CONV_WIDTH = 3
NUM_OUTPUTS = 1
MEMORY_UNITS = 32


#############################################################################
class SingleStepLastValueBaseLine(kerasModel):
    def __init__(self, output_column_id=None):
        super().__init__()
        self.output_column_id = output_column_id

    def call(self, inputs):
        if self.output_column_id is None:
            return inputs
        result = inputs[:, :, self.output_column_id]
        return result[:, :, tf.newaxis]


class SingleStepLinear(kerasModel):

    def __init__(self):
        super().__int__()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=1)
        ])

    def call(self, inputs):
        outputs = self.net(inputs)
        return outputs

    def plot_weights(self, feature_names):
        plt.bar(x=range(len(feature_names)),
                height=self.layers[0].kernel[:, 0].numpy())
        axis = plt.gca()
        axis.set_xticks(range(len(feature_names)))
        _ = axis.set_xticklabels(feature_names, rotation=90)
        plt.title('Weights for each input variable')
        plt.show()


class SingleStepDense(kerasModel):

    def __init__(self):
        super().__init__()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=1)
        ])

    def call(self, inputs):
        outputs = self.net(inputs)
        return outputs


class SingleStepFlattenedDense(kerasModel):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.models.Sequential([
            # Shape: (time, features) => (time*features)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1),
            # Add back the time dimension.
            # Shape: (outputs) => (1, outputs)
            tf.keras.layers.Reshape([1, -1]),
        ])

    def call(self, inputs):
        outputs = self.net(inputs)
        return outputs


class SingleStepCNN(kerasModel):
    def __init__(self, conv_width=CONV_WIDTH):
        super().__init__()
        self.conv_width = conv_width
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(filters=32,
                                   kernel_size=(conv_width,),
                                   activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1),
        ])

    def call(self, inputs):
        outputs = self.net(inputs)
        return outputs


class SingleStepLSTM(kerasModel):
    def __init__(self, units=MEMORY_UNITS):
        super().__init__()
        self.units = units
        self.net = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.LSTM(units, return_sequences=True),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(units=1)
        ])

    def call(self, inputs):
        outputs = self.net(inputs)
        return outputs


class SingleStepGRU(kerasModel):
    def __init__(self, units=MEMORY_UNITS):
        super().__init__()
        self.units = units
        self.net = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, time, lstm_units]
            tf.keras.layers.GRU(units, return_sequences=True),
            # Shape => [batch, time, features]
            tf.keras.layers.Dense(units=1)
        ])

    def call(self, inputs):
        outputs = self.net(inputs)
        return outputs


class MultiStepLastValueBaseline(kerasModel):
    def __init__(self, prediction_length=OUT_STEPS, output_column_id=-1):
        super().__init__()
        self.output_column_id = output_column_id
        self.prediction_length = prediction_length

    def call(self, inputs):
        outputs = tf.tile(inputs[:, -1:, self.output_column_id:], [1, self.prediction_length, 1])
        return outputs


class MultiStepRepeatBaseline(kerasModel):
    def __init__(self, output_column_id=-1):
        super().__init__()
        self.output_column_id = output_column_id

    def call(self, inputs):
        return inputs[:, :, self.output_column_id:]


class MultiStepLinear(kerasModel):
    def __init__(self, prediction_length=OUT_STEPS, num_output_features=NUM_OUTPUTS):
        super().__init__()
        self.num_output_features = num_output_features
        self.prediction_length = prediction_length
        self.net = tf.keras.models.Sequential([
            # Take the last time-step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, out_steps*features]
            tf.keras.layers.Dense(prediction_length * num_output_features, kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([prediction_length, num_output_features])
        ])

    def call(self, inputs):
        outputs = self.net(inputs)
        return outputs


class MultiStepDense(kerasModel):
    def __init__(self, prediction_length=OUT_STEPS, num_output_features=NUM_OUTPUTS):
        super().__init__()
        self.num_output_features = num_output_features
        self.prediction_length = prediction_length
        self.net = tf.keras.models.Sequential([
            # Take the last time step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, dense_units]
            tf.keras.layers.Dense(512, activation='relu'),
            # Shape => [batch, out_steps*features]
            tf.keras.layers.Dense(prediction_length * num_output_features,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([prediction_length, num_output_features])
        ])

    def call(self, inputs):
        outputs = self.net(inputs)
        return outputs


class MultiStepCNN(kerasModel):
    def __init__(self, prediction_length=OUT_STEPS, conv_width=CONV_WIDTH, num_output_features=NUM_OUTPUTS):
        super().__init__()
        self.num_features = num_output_features
        self.conv_width = conv_width
        self.prediction_length = prediction_length
        self.net = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
            tf.keras.layers.Lambda(lambda x: x[:, -conv_width:, :]),
            # Shape => [batch, 1, conv_units]
            tf.keras.layers.Conv1D(256, activation='relu', kernel_size=conv_width),
            # Shape => [batch, 1,  out_steps*features]
            tf.keras.layers.Dense(prediction_length * num_output_features,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([prediction_length, num_output_features])
        ])

    def call(self, inputs):
        outputs = self.net(inputs)
        return outputs


class MultiStepLSTM(kerasModel):
    def __init__(self, prediction_length=OUT_STEPS, num_output_features=NUM_OUTPUTS, lstm_units=32):
        super().__init__()
        self.prediction_length = prediction_length
        self.num_output_features = num_output_features
        self.lstm_units = lstm_units
        self.net = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(lstm_units, return_sequences=False),
            # Shape => [batch, out_steps*features].
            tf.keras.layers.Dense(prediction_length * num_output_features, kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([prediction_length, num_output_features])
        ])

    def call(self, inputs):
        outputs = self.net(inputs)
        return outputs


class MultiStepMemoryFeedback(kerasModel):
    def __init__(self, units, out_steps, num_features, output_column_id, memory_unit=tf.keras.layers.LSTMCell):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.num_features = num_features
        self.output_column_id = output_column_id
        self.memory_cell = memory_unit(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.memory_rnn = tf.keras.layers.RNN(self.memory_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.memory_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def show_dimensions(self, multi_window):
        prediction, state = self.warmup(multi_window.example[0])
        print(f' Prediction shape {prediction.shape}')
        print('Output shape (batch, time, features): ', self.call(multi_window.example[0]).shape)

    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)

        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.memory_cell(x, states=state,
                                        training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        predictions = predictions[:, :, self.output_column_id:]

        return predictions


#############################################################################
def make_single_step_model(model_name, conv_width=CONV_WIDTH, output_column_id=None):
    if model_name == 'baseline':
        model = SingleStepLastValueBaseLine(output_column_id=output_column_id)
        descr = 'Baseline model (Prediction is last value)'
    elif model_name == 'linear':
        model = SingleStepLinear()
        descr = 'Linear model'
    elif model_name == 'dense' or model_name == 'fc':
        model = SingleStepDense()
        descr = 'FC model'
    elif model_name == 'flattened dense':
        model = SingleStepFlattenedDense()
        descr = 'FC model with lookback'
    elif model_name == 'cnn':
        model = SingleStepCNN(conv_width=conv_width)
        descr = 'Convolutional model'
    elif model_name == 'lstm':
        model = SingleStepLSTM(units=32)
        descr = 'LSTM model'
    elif model_name == 'gru':
        model = SingleStepGRU(units=32)
        descr = 'GRU model'
    else:
        model = None
        descr = 'None'

    return model, descr


def make_multi_step_model(model_name, prediction_length=OUT_STEPS, num_output_features=NUM_OUTPUTS,
                          conv_width=CONV_WIDTH, memory_units=MEMORY_UNITS, output_column_id=-1, num_features=1):
    if model_name == 'lastbaseline':
        model = MultiStepLastValueBaseline(prediction_length, output_column_id=output_column_id)
        descr = 'Baseline model (Prediction is last value)'
    elif model_name == 'repeatbaseline':
        model = MultiStepRepeatBaseline(output_column_id=output_column_id)
        descr = 'Baseline model (Prediction equals past sequence of values)'
    elif model_name == 'linear':
        model = MultiStepLinear(prediction_length, num_output_features)
        descr = 'Linear model'
    elif model_name == 'dense' or model_name == 'fc':
        model = MultiStepDense(prediction_length, num_output_features)
        descr = 'FC model'
    elif model_name == 'cnn':
        model = MultiStepCNN(prediction_length, conv_width, num_output_features)
        descr = 'Convolutional model'
    elif model_name == 'lstm':
        model = MultiStepLSTM(prediction_length, num_output_features, lstm_units=memory_units)
        descr = 'LSTM model'
    elif model_name == 'arlstm':
        memory_unit = tf.keras.layers.LSTMCell
        model = MultiStepMemoryFeedback(units=memory_units, out_steps=prediction_length,
                                        num_features=num_features, output_column_id=output_column_id,
                                        memory_unit=memory_unit)
        descr = 'Autoregressive Feedback model'
    else:
        model = None
        descr = 'None'

    return model, descr


def compile_and_fit(model, window, patience=2, max_epochs=MAX_EPOCHS):
    early_stopping = tfEarlyStopping(monitor='val_loss',
                                     patience=patience,
                                     mode='min')

    model.compile(loss=MeanSquaredError(),
                  optimizer=Adam(),
                  metrics=[MeanAbsoluteError()])

    history = model.fit(window.train, epochs=max_epochs, verbose=0, validation_data=window.val,
                        callbacks=[early_stopping])

    # IPython.display.clear_output()

    return history


def compile_and_evaluate(model, train_window_generator, plot_window_generator=None, target_column=None, descr=None,
                         max_epochs=MAX_EPOCHS):
    if descr is not None:
        print(f'\n\n{descr}\n\n')

    history = compile_and_fit(model, train_window_generator, max_epochs=max_epochs)
    # model.compile(loss=MeanSquaredError(),
    #                 metrics=[MeanAbsoluteError()])

    val_perf = model.evaluate(train_window_generator.val)
    test_perf = model.evaluate(train_window_generator.test, verbose=0)

    if plot_window_generator is not None and target_column is not None:
        plot_window_generator.plot(model, plot_col=target_column)

    return model, val_perf, test_perf, history


def run_single_step_model(model_name, single_step_window, wide_window, target_column=None, output_column_id=None,
                          conv_width=CONV_WIDTH, max_epochs=MAX_EPOCHS):
    model, descr = make_single_step_model(model_name, conv_width, output_column_id)
    model, val_perf, test_perf, history = compile_and_evaluate(model, single_step_window, wide_window, target_column,
                                                               descr, max_epochs)
    return model, val_perf, test_perf, history


def run_multi_step_model(model_name, multi_window, prediction_length=OUT_STEPS, num_output_features=1, target_column=None,
                         output_column_id=-1, conv_width=CONV_WIDTH, memory_units=MEMORY_UNITS, num_features=1,
                         max_epochs=MAX_EPOCHS):
    model, descr = make_multi_step_model(model_name, prediction_length, num_output_features, conv_width, memory_units,
                                         output_column_id, num_features)
    model, val_perf, test_perf, history = compile_and_evaluate(model, multi_window, multi_window, target_column,
                                                               descr, max_epochs)
    return model, val_perf, test_perf, history

