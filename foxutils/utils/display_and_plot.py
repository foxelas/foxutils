import matplotlib
import matplotlib.pyplot as plt

import matplotlib.dates as mdates
from sklearn.metrics import ConfusionMatrixDisplay

import geopandas as gpd
from shapely.geometry import Point

import folium
from folium import plugins


# matplotlib.use('Qt5Agg')


def print_timeseries(timestamps, values, plot_title):
    fig, ax = plt.subplots()
    ax.plot(timestamps, values)
    ax.set_xlabel("time points")
    ax.set_ylabel("values")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')

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


def plot_coords_on_map(streetmap_file, df, label_column='ID', crs='epsg:4326'):
    street_map = gpd.read_file(streetmap_file)
    long = df['Longitude'].values
    lat = df['Latitude'].values
    geometry = [Point(xy) for xy in zip(long, lat)]
    geo_df = gpd.GeoDataFrame(df, geometry=geometry)
    geo_df.crs = crs

    fig, ax = plt.subplots(figsize=(15, 15))
    street_map.plot(ax=ax, alpha=0.4, color='grey')
    geo_df.plot(ax=ax, markersize=20, color='blue', marker='o')
    geo_df.apply(lambda x: ax.annotate(x[label_column], xy=x.loc['geometry'].coords[0]), axis=1)
    # plt.legend(prop={'size':15})


def plot_markers_on_map(center_coords, df, label_column='ID'):
    m = folium.Map(location=center_coords)

    long = df['Longitude'].values
    lat = df['Latitude'].values
    labels = df[label_column].values

    for (x,y, z) in zip(lat, long, labels):
        folium.Marker(
            location=[x,y],
            popup=(label_column + ': ' + str(z)),  # pop-up label for the marker
            icon=folium.Icon()
        ).add_to(m)

    return m
