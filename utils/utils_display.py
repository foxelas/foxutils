import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import matplotlib.dates as mdates
from sklearn.metrics import ConfusionMatrixDisplay


def print_timeseries(timestamps, values, plot_title):
    fig, ax = plt.subplots()
    ax.plot(timestamps, values)
    ax.set_xlabel("time points")
    ax.set_ylabel("values")
    ax.title.set_text(plot_title)
    plt.show()


def print_subplots_for_btc_data(data_df, plot_title):
    rates = data_df['rate']
    timestamps = data_df['timestamp']
    open_values = data_df['current_open']
    close_values = data_df['future_close']

    fig, axs = plt.subplots(1, 3, figsize=(20, 5))

    ax = axs[0]
    ax.plot(timestamps, rates)
    ax.set_xlabel("time points")
    ax.set_ylabel("rates")
    ax.title.set_text("BTC/USD rates")

    ax = axs[1]
    ax.plot(timestamps, open_values)
    ax.set_xlabel("time points")
    ax.set_ylabel("current open price")
    ax.title.set_text("BTC/USD current open")

    ax = axs[2]
    ax.plot(timestamps, close_values)
    ax.set_xlabel("time points")
    ax.set_ylabel("future close price")
    ax.title.set_text("BTC/USD future close")

    for ax in axs:
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%b"))
        for label in ax.get_xticklabels(which='major'):
            label.set(rotation=30, horizontalalignment='right')

    plt.suptitle(plot_title)
    plt.show()


def plot_confusion_matrix(confusion_matrix, model_name):
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
    cm_display.plot()
    plt.title(model_name)
    plt.show()


def plot_value_per_epoch(train_vals, test_vals, target_value, plot_title="", legend_names=None):
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(train_vals)
    ax.plot(test_vals)
    plt.ylabel(target_value)
    plt.xlabel('Epoch')
    if legend_names is None:
        plt.legend(['train', 'valid'], loc='upper left')
    else:
        plt.legend(legend_names, loc='upper left')

    ax.title.set_text(plot_title)
    plt.show()


def plot_accuracy_per_epoch(train_vals, test_vals, plot_title="Accuracy per Epoch", legend_names=None):
    plot_value_per_epoch(train_vals, test_vals, "Accuracy", plot_title, legend_names)


def plot_loss_per_epoch(train_vals, test_vals, plot_title="Loss per Epoch", legend_names=None):
    plot_value_per_epoch(train_vals, test_vals, "Loss", plot_title, legend_names)
