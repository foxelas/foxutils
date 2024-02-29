import numpy as np
from os.path import join as pathjoin
import matplotlib.pyplot as plt


# matplotlib.use('Qt5Agg')


def plot_histogram(df, target_column, category_dict=None, plot_title='', print_texts=False):
    import matplotlib.pyplot as plt

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
    import matplotlib.dates as mdates

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
    import matplotlib.dates as mdates

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


def plot_confusion_matrix(confusion_matrix, title=None,
                          package="sklearn", class_names=None,
                          show_plot=True, **params):
    if package == "sklearn":
        from sklearn.metrics import ConfusionMatrixDisplay

        params = dict(confusion_matrix=confusion_matrix)
        if class_names is not None:
            params.update(display_labels=class_names)

        cm_display = ConfusionMatrixDisplay(**params)
        fig_ = cm_display.plot()

    elif package == "mlxtend":
        from mlxtend.plotting import plot_confusion_matrix

        if params is None:
            params = dict(
                          colorbar=True,
                          show_absolute=True,
                          show_normed=True,
                          )
        if class_names is not None:
            params.update(class_names=class_names)

        fig_, ax = plot_confusion_matrix(conf_mat=confusion_matrix, **params)

    elif package == "lightning":
        import seaborn as sns
        import pandas as pd

        df_cm = pd.DataFrame(confusion_matrix)
        if class_names:
            if not isinstance(class_names, dict):
                class_names = {k: i for i, k in enumerate(class_names)}
            inv_map = {v: k for k, v in class_names.items()}
            df_cm.rename(columns=inv_map, index=inv_map, inplace=True)

        fig_ = sns.heatmap(df_cm, annot=True, cmap="Spectral", fmt="d").get_figure()

    else:
        raise ValueError("Value should be one of sklearn, lightning or mlxtend.")

    if title:
        plt.title(title)

    if show_plot:
        plt.show()

    return fig_


def plot_value_per_epoch(train_vals, test_vals, target_value, plot_title="", legend_names=None):
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(train_vals)
    ax.plot(test_vals)
    plt.ylabel(target_value)
    plt.xlabel("Epoch")
    if legend_names is None:
        plt.legend(["train", "valid"], loc="upper left")
    else:
        plt.legend(legend_names, loc="upper left")

    ax.title.set_text(plot_title)
    plt.show()


def plot_accuracy_per_epoch(train_vals, test_vals, plot_title="Accuracy per Epoch", legend_names=None):
    plot_value_per_epoch(train_vals, test_vals, "Accuracy", plot_title, legend_names)


def plot_loss_per_epoch(train_vals, test_vals, plot_title="Loss per Epoch", legend_names=None):
    plot_value_per_epoch(train_vals, test_vals, "Loss", plot_title, legend_names)


def plot_coords_on_map(streetmap_file, df, label_column="ID", crs="epsg:4326"):
    import geopandas as gpd
    from shapely.geometry import Point

    street_map = gpd.read_file(streetmap_file)
    lng = df["lng"].values
    lat = df["lat"].values
    geometry = [Point(xy) for xy in zip(lat, lng)]
    geo_df = gpd.GeoDataFrame(df, geometry=geometry)
    geo_df.crs = crs

    fig, ax = plt.subplots(figsize=(15, 15))
    street_map.plot(ax=ax, alpha=0.4, color="grey")
    geo_df.plot(ax=ax, markersize=20, color="blue", marker="o")
    geo_df.apply(lambda x: ax.annotate(x[label_column], xy=x.loc["geometry"].coords[0]), axis=1)
    # plt.legend(prop={'size':15})


def plot_markers_on_map(center_coords=None, df=None, m=None, label_column="ID", color="blue"):
    import folium
    from folium import plugins

    if m is None:
        if center_coords is None and df is None:
            m = folium.Map()
        else:
            if center_coords is None:
                center_coords = [df.iloc[0]["lat"], df.iloc[0]["lng"]]
            m = folium.Map(location=center_coords)

    lng = df["lng"].values
    lat = df["lat"].values
    labels = df[label_column].values

    for (x, y, z) in zip(lat, lng, labels):
        folium.Marker(
            location=[x, y],
            popup=(label_column + ": " + str(z)),  # pop-up label for the marker
            icon=folium.Icon(color=color)
        ).add_to(m)

    return m


def display_first_frames_from_h264(filedir, filename):
    from PIL import Image
    import cv2
    import av

    if not (".h264" in filename):
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
