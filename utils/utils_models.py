import tensorflow as tf
from tensorflow.keras import Model as kerasModel
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError

from .utils_train import compile_and_fit
import matplotlib.pyplot as plt

CONV_WIDTH = 3
NUM_OUTPUTS = 1
MAX_EPOCHS = 20


class Baseline(kerasModel):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


def run_baseline(single_step_window, wide_window, target_column, target_column_id=-1):
    print('\n\nBaseline model\n')

    baseline = Baseline(label_index=target_column_id)
    baseline.compile(loss=MeanSquaredError(),
                     metrics=[MeanAbsoluteError()])

    val_perf = baseline.evaluate(single_step_window.val)
    test_perf = baseline.evaluate(single_step_window.test, verbose=0)

    wide_window.plot(baseline, plot_col=target_column)
    return baseline, val_perf, test_perf


def run_linear(single_step_window, wide_window, feature_names, target_column, max_epochs=MAX_EPOCHS):
    print('\n\nLinear model\n')

    linear = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=1)
    ])
    history = compile_and_fit(linear, single_step_window, max_epochs=max_epochs)
    val_perf = linear.evaluate(single_step_window.val)
    test_perf = linear.evaluate(single_step_window.test, verbose=0)
    wide_window.plot(linear, plot_col=target_column)

    plt.bar(x=range(len(feature_names)),
            height=linear.layers[0].kernel[:, 0].numpy())
    axis = plt.gca()
    axis.set_xticks(range(len(feature_names)))
    _ = axis.set_xticklabels(feature_names, rotation=90)
    plt.title('Weights for each input variable')
    plt.show()

    return linear, val_perf, test_perf, history


def run_fc(single_step_window, wide_window, target_column, max_epochs=MAX_EPOCHS):
    print('\n\nFC model\n')

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])
    history = compile_and_fit(model, single_step_window, max_epochs=max_epochs)
    val_perf = model.evaluate(single_step_window.val)
    test_perf = model.evaluate(single_step_window.test, verbose=0)
    wide_window.plot(model, plot_col=target_column)
    return model, val_perf, test_perf, history


def run_multistep_fc(conv_window_gen, target_column, max_epochs=MAX_EPOCHS):
    print('\n\nMultistep FC model\n')

    multi_step_dense = tf.keras.models.Sequential([
        # Shape: (time, features) => (time*features)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
        # Add back the time dimension.
        # Shape: (outputs) => (1, outputs)
        tf.keras.layers.Reshape([1, -1]),
    ])
    history = compile_and_fit(multi_step_dense, conv_window_gen, max_epochs=max_epochs)

    val_perf = multi_step_dense.evaluate(conv_window_gen.val)
    test_perf = multi_step_dense.evaluate(conv_window_gen.test, verbose=0)
    conv_window_gen.plot(multi_step_dense, plot_col=target_column)
    return multi_step_dense, val_perf, test_perf, history


def run_cnn(conv_window_gen, wide_conv_window, target_column, conv_width=CONV_WIDTH, max_epochs=MAX_EPOCHS):
    print('\n\nCNN model\n')

    conv_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=(conv_width,),
                               activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
    ])

    history = compile_and_fit(conv_model, conv_window_gen, max_epochs=max_epochs)
    val_perf = conv_model.evaluate(conv_window_gen.val)
    test_perf = conv_model.evaluate(conv_window_gen.test, verbose=0)
    wide_conv_window.plot(conv_model, plot_col=target_column)
    return conv_model, val_perf, test_perf, history


def run_lstm(conv_window_gen, wide_window, target_column, max_epochs=MAX_EPOCHS):
    print('\n\nLSTM model\n')

    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])

    history = compile_and_fit(lstm_model, conv_window_gen, max_epochs=max_epochs)
    val_perf = lstm_model.evaluate(conv_window_gen.val)
    test_perf = lstm_model.evaluate(conv_window_gen.test, verbose=0)
    wide_window.plot(lstm_model, plot_col=target_column)
    return lstm_model, val_perf, test_perf, history


def run_gru(conv_window_gen, wide_window, target_column, max_epochs=MAX_EPOCHS):
    print('\n\nGRU model\n')

    gru_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.GRU(32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])

    history = compile_and_fit(gru_model, conv_window_gen, max_epochs=max_epochs)
    val_perf = gru_model.evaluate(conv_window_gen.val)
    test_perf = gru_model.evaluate(conv_window_gen.test, verbose=0)
    wide_window.plot(gru_model, plot_col=target_column)
    return gru_model, val_perf, test_perf, history


class MultiStepLastBaseline(kerasModel):
    def __init__(self, prediction_length, output_column_id=-1):
        super().__init__()
        self.output_column_id = output_column_id
        self.prediction_length = prediction_length

    def call(self, inputs):
        outputs = tf.tile(inputs[:, -1:, self.output_column_id:], [1, self.prediction_length, 1])
        return outputs


class RepeatBaseline(kerasModel):
    def __init__(self, output_column_id=-1):
        super().__init__()
        self.output_column_id = output_column_id

    def call(self, inputs):
        return inputs[:, :, self.output_column_id:]


class Multi_Linear_Model(kerasModel):
    def __init__(self, prediction_length, output_features=NUM_OUTPUTS):
        super().__init__()
        self.output_features = output_features
        self.prediction_length = prediction_length
        self.net = tf.keras.models.Sequential([
            # Take the last time-step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, out_steps*features]
            tf.keras.layers.Dense(prediction_length * output_features, kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([prediction_length, output_features])
        ])

    def call(self, inputs):
        outputs = self.net(inputs)
        return outputs


class Multi_Dense_Model(kerasModel):
    def __init__(self, prediction_length, output_features=NUM_OUTPUTS):
        super().__init__()
        self.output_features = output_features
        self.prediction_length = prediction_length
        self.net = tf.keras.models.Sequential([
            # Take the last time step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, dense_units]
            tf.keras.layers.Dense(512, activation='relu'),
            # Shape => [batch, out_steps*features]
            tf.keras.layers.Dense(prediction_length * output_features,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([prediction_length, output_features])
        ])

    def call(self, inputs):
        outputs = self.net(inputs)
        return outputs


class Multi_Conv_Model(kerasModel):
    def __init__(self, prediction_length, conv_width=CONV_WIDTH, output_features=NUM_OUTPUTS):
        super().__init__()
        self.num_features = output_features
        self.conv_width = conv_width
        self.prediction_length = prediction_length
        self.net = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
            tf.keras.layers.Lambda(lambda x: x[:, -conv_width:, :]),
            # Shape => [batch, 1, conv_units]
            tf.keras.layers.Conv1D(256, activation='relu', kernel_size=conv_width),
            # Shape => [batch, 1,  out_steps*features]
            tf.keras.layers.Dense(prediction_length * output_features,
                                  kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([prediction_length, output_features])
        ])

    def call(self, inputs):
        outputs = self.net(inputs)
        return outputs


class Multi_LSTM_Model(kerasModel):
    def __init__(self, prediction_length, output_features=NUM_OUTPUTS, lstm_units=32):
        super().__init__()
        self.prediction_length = prediction_length
        self.output_features = output_features
        self.lstm_units = lstm_units
        self.net = tf.keras.models.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(lstm_units, return_sequences=False),
            # Shape => [batch, out_steps*features].
            tf.keras.layers.Dense(prediction_length * output_features, kernel_initializer=tf.initializers.zeros()),
            # Shape => [batch, out_steps, features].
            tf.keras.layers.Reshape([prediction_length, output_features])
        ])

    def call(self, inputs):
        outputs = self.net(inputs)
        return outputs


class FeedBack(kerasModel):
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


def run_multi_step_last_value_baseline(multi_window, prediction_length, target_column, output_column_id):
    print('\n\nBaseline model: Prediction equals last value\n')
    last_baseline = MultiStepLastBaseline(prediction_length, output_column_id=output_column_id)
    last_baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                          metrics=[tf.keras.metrics.MeanAbsoluteError()])

    val_perf = last_baseline.evaluate(multi_window.val)
    test_perf = last_baseline.evaluate(multi_window.test, verbose=0)

    multi_window.plot(last_baseline, plot_col=target_column)
    return last_baseline, val_perf, test_perf


def run_repeat_segment_baseline(multi_window, target_column, output_column_id):
    print('\n\nBaseline model: Prediction equals past sequence of values\n')

    model = RepeatBaseline(output_column_id=output_column_id)
    model.compile(loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError()])
    val_perf = model.evaluate(multi_window.val)
    test_perf = model.evaluate(multi_window.test, verbose=0)
    multi_window.plot(model, plot_col=target_column)
    return model, val_perf, test_perf


def run_multi_linear_model(multi_window, prediction_length, output_features, target_column, max_epochs=MAX_EPOCHS):
    print('\n\nLinear model\n')

    model = Multi_Linear_Model(prediction_length, output_features)
    history = compile_and_fit(model, multi_window, max_epochs=max_epochs)
    val_perf = model.evaluate(multi_window.val)
    test_perf = model.evaluate(multi_window.test, verbose=0)
    multi_window.plot(model, plot_col=target_column)
    return model, val_perf, test_perf, history


def run_multi_dense_model(multi_window, prediction_length, output_features, target_column, max_epochs=MAX_EPOCHS):
    print('\n\nFC model\n')

    model = Multi_Dense_Model(prediction_length, output_features)

    history = compile_and_fit(model, multi_window, max_epochs=max_epochs)
    val_perf = model.evaluate(multi_window.val)
    test_perf = model.evaluate(multi_window.test, verbose=0)
    multi_window.plot(model, plot_col=target_column)
    return model, val_perf, test_perf, history


def run_multi_conv_model(multi_window, prediction_length, output_features, conv_width, target_column,
                         max_epochs=MAX_EPOCHS):
    print('\n\nCNN model\n')
    model = Multi_Conv_Model(prediction_length, conv_width, output_features)

    history = compile_and_fit(model, multi_window, max_epochs=max_epochs)
    val_perf = model.evaluate(multi_window.val)
    test_perf = model.evaluate(multi_window.test, verbose=0)
    multi_window.plot(model, plot_col=target_column)
    return model, val_perf, test_perf, history


def run_multi_lstm_model(multi_window, prediction_length, output_features, target_column, lstm_units=32,
                         max_epochs=MAX_EPOCHS):
    print('\n\nLSTM model\n')

    model = Multi_LSTM_Model(prediction_length, output_features, lstm_units=lstm_units)

    history = compile_and_fit(model, multi_window, max_epochs=max_epochs)
    val_perf = model.evaluate(multi_window.val)
    test_perf = model.evaluate(multi_window.test, verbose=0)
    multi_window.plot(model, plot_col=target_column)
    return model, val_perf, test_perf, history


def run_autoregressive_feedback_model(multi_window, target_column, output_column_id, prediction_length, num_features,
                                      units=32, memory_unit=tf.keras.layers.LSTMCell, max_epochs=MAX_EPOCHS):
    print('\n\nAutoregressive Feedback model\n')

    feedback_model = FeedBack(units=units, out_steps=prediction_length, num_features=num_features,
                              output_column_id=output_column_id, memory_unit=memory_unit)

    prediction, state = feedback_model.warmup(multi_window.example[0])
    print(f' Prediction shape {prediction.shape}')
    print('Output shape (batch, time, features): ', feedback_model(multi_window.example[0]).shape)

    history = compile_and_fit(feedback_model, multi_window, max_epochs=max_epochs)

    val_perf = feedback_model.evaluate(multi_window.val)
    test_perf = feedback_model.evaluate(multi_window.test, verbose=0)
    multi_window.plot(feedback_model, plot_col=target_column)
    return feedback_model, val_perf, test_perf, history
