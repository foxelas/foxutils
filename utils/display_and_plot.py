import matplotlib.pyplot as plt

import matplotlib.dates as mdates
from sklearn.metrics import ConfusionMatrixDisplay

import av
import cv2
import numpy as np
from PIL import Image


from os.path import join as pathjoin

# matplotlib.use('Qt5Agg')


def plot_histogram(df, target_column, category_dict=None, plot_title='', print_texts=False):
    if print_texts:
        print(f"\nNumber of entries: {len(df)}")
        print(f"Unique categories:\n {np.unique(df[target_column].dropna().values)}")

    hist = df[target_column].value_counts()
    # hist.sort_values(inplace=True, ascending=False)

    if category_dict is None:
        category_dict = {k: k for k in hist.index}

    inv_map = {v: k for k, v in category_dict.items()}
    v = [inv_map[x] for x in hist.index]
    plt.bar(v, hist, width=0.4)
    plt.xticks(rotation='vertical')
    plt.title(plot_title)
    plt.show()

    if print_texts:
        print(hist)

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
    import geopandas as gpd
    from shapely.geometry import Point

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

    import folium
    from folium import plugins

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





def display_first_frames_from_h264(filedir, filename):
    if not '.h264' in filename:
        filename = filename + ".h264"

    file = pathjoin(filedir, filename)
    print(f'\nReading from {file}\n')

    container = av.open(file)
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_image())
        if len(frames) == 5:
            break

    for frame in frames:
        img = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        im_pil = Image.fromarray(img)
        im_pil.show()


