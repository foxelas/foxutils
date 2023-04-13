from os.path import join as pathjoin
from os.path import normpath, isfile
from os import listdir, remove
import configparser
from sys import getsizeof
import json
import h5py
import pandas as pd
import gzip
import tarfile
import torch

from os import getcwd
from datetime import datetime, date, timedelta

###########################################################
# The filename of the settings file
settings_filename = 'config.ini'


# Reads settings from file
def read_config():
    config = configparser.ConfigParser()
    config.read(settings_filename)
    #print(getcwd())
    #print(config.sections())
    return config


settings = read_config()

lookback_window = int(settings['RUN']['lookback_window'])
use_description = bool(eval(settings['RUN']['use_description']))
fixed_length_historical_sentiments = int(settings['RUN']['fixed_length_historical_sentiments'])
default_value_sentiment = float(settings['RUN']['default_value_sentiment'])
target_features_value = settings['RUN']['target_features_value']
output_threshold = float(settings['RUN']['output_threshold'])
influence_hours = int(settings['RUN']['influence_hours'])
target_dataset = settings['RUN']['target_dataset']
disable_lemmatizer = bool(eval(settings['RUN']['disable_lemmatizer']))

datasets_dir = normpath(settings['DIRECTORY']['datasets_dir'])
models_dir = normpath(settings['DIRECTORY']['models_dir'])
token_dir = normpath(settings['DIRECTORY']['token_dir'])
preprocessed_folder = settings['DIRECTORY']['preprocessed_folder']
extracted_folder = settings['DIRECTORY']['extracted_folder']

dataset_filename = settings['HDF']['dataset_filename'] #pathjoin(datasets_dir, target_dataset, settings['HDF']['dataset_filename'])
fetched_data_filename = settings['HDF']['fetched_data_filename'] #pathjoin(datasets_dir, target_dataset, settings['HDF']['fetched_data_filename'])
newsapi_group = settings['HDF']['newsapi_group']
yfinance_group = settings['HDF']['yfinance_group']

news_group = settings['HDF']['news_group']
asset_group = settings['HDF']['asset_group']
preprocessed_group = settings['HDF']['preprocessed_group']
performance_group = settings['HDF']['performance_group']
preprocessed_asset_group = settings['HDF']['preprocessed_asset_group']

sentiment_analysis_model = settings['MODELS']['sentiment_analysis_model']
sentence_transformer_model = settings['MODELS']['sentence_transformer_model']
summary_model = settings['MODELS']['summary_model']

batch_size = int(settings['SENTIMENT']['batch_size'])
has_shuffle = bool(eval(settings['SENTIMENT']['has_shuffle']))
num_workers = int(settings['SENTIMENT']['num_workers'])

is_test = settings['RUN']['is_test']
test_suffix = '_test' if is_test else ''

encoding = "utf-8"

h5File = pathjoin(datasets_dir, target_dataset, settings['HDF']['dataset_filename'])

if __name__ == '__main__':
    print(f"\n--------------------Parameters used in this run--------------------")
    print(f"Target dataset: {target_dataset}")
    print(f"Lookback window: {lookback_window} hours")
    print(f"Influence window: {influence_hours} hours")
    print(f"Use description: {use_description}")
    print(f"output_threshold: {output_threshold}")
    print(f"fixed_length_historical_sentiments: {fixed_length_historical_sentiments}")
    print(f"default_value_sentiment: {default_value_sentiment}")
    print(f"target_features_value: {target_features_value}")

###########################################################
# Tokens for API access
def get_api_key(filename):
    token_data = json.load(open(pathjoin(token_dir, filename)))
    token = token_data.get("key")
    return token


###########################################################
# Datetimes
def convert_string_to_date(date_string):
    try:
        date_value = datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ")
    except:
        try:
            date_value = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
        except:
            date_value = datetime.strptime(date_string, "%Y-%m-%d %I-%p")

    return date_value


def convert_date_to_string(date_value):
    date_value = datetime.strftime(date_value, "%Y-%m-%d")
    return date_value


def convert_datetime_to_string(date_value):
    date_value = datetime.strftime(date_value, "%Y-%m-%dT%H:%M:%SZ")
    return date_value


def get_datetime_from_unix_timestamp(timestamp):
    date_value = datetime.fromtimestamp(timestamp / 1000)
    return date_value


###########################################################
# Memory

def get_device():
    device_ = "cuda" if torch.cuda.is_available() else "cpu"
    if __name__ == '__main__':
        print(f'Running on {device_}')
    return device_


device = get_device()


def obj_size_fmt(num):
    if num < 10 ** 3:
        return "{:.2f}{}".format(num, "B")
    elif (num >= 10 ** 3) & (num < 10 ** 6):
        return "{:.2f}{}".format(num / (1.024 * 10 ** 3), "KB")
    elif (num >= 10 ** 6) & (num < 10 ** 9):
        return "{:.2f}{}".format(num / (1.024 * 10 ** 6), "MB")
    else:
        return "{:.2f}{}".format(num / (1.024 * 10 ** 9), "GB")


def memory_usage():
    memory_usage_by_variable = pd.DataFrame({k: getsizeof(v) for (k, v) in globals().items()}, index=['Size'])
    memory_usage_by_variable = memory_usage_by_variable.T
    memory_usage_by_variable = memory_usage_by_variable.sort_values(by='Size', ascending=False).head(10)
    memory_usage_by_variable['Size'] = memory_usage_by_variable['Size'].apply(lambda x: obj_size_fmt(x))
    return memory_usage_by_variable


def delete_files_by_extension(target_folder, target_extension):
    for filename in listdir(target_folder):
        # Check file extension
        if filename.endswith(target_extension):
            # Remove File
            remove(pathjoin(target_folder, filename))


###########################################################
# JSON files
def get_item_from_json_file(data, key):
    if key in data.keys():
        return data[key]
    else:
        return ''


def save_data_to_json_gz(filename, data):
    json_str = json.dumps(data)
    json_bytes = json_str.encode('utf-8')
    with gzip.open(filename, 'w') as outfile:
        outfile.write(json_bytes)


def load_data_from_json_gz(filename):
    # also works with file object instead of filename
    with gzip.open(filename, mode="r") as f:
        data = json.loads(f.read().decode(encoding=encoding))
        return data


###########################################################
# TAR files
def compress_files_to_tar(filename, filelist, file_dir, remove_extension=None):
    with tarfile.open(filename, 'w') as tar:
        for target_file in filelist:
            tar.add(pathjoin(file_dir, target_file), arcname=target_file)

    if remove_extension:
        delete_files_by_extension(file_dir, remove_extension)


def extract_files_from_tar(file_dir, save_dir):
    data_files = listdir(file_dir)
    data_files = [i for i in data_files if ('.tar' in i)]

    for tar_file in data_files:
        filename = pathjoin(file_dir, tar_file)
        tar = tarfile.open(filename)
        tar.extractall(save_dir)
        tar.close()


###########################################################
# H5 files
def all_keys(obj):
    keys = (obj.name,)
    if isinstance(obj, h5py.Group):
        for key, value in obj.items():
            if isinstance(value, h5py.Group):
                keys = keys + all_keys(value)
            else:
                keys = keys + (value.name,)
    return keys


def print_keys_of_h5(filename):
    with h5py.File(filename, "r") as f:
        print(all_keys(f))


def delete_keys_of_h5(filename, name):
    with h5py.File(filename, "a") as f:
        del f[name]


def print_groups_of_h5(filename):
    data = get_groups_of_h5(filename)
    print(data)


def join_h5_keys(values):
    return "/".join(values)


def get_groups_of_h5(filename):
    with h5py.File(filename, "r") as f:
        data = ["/" + x + "/" + y for x in list(f.keys()) for y in list(f[x])]
        return data


def split_and_convert_to_int(value):
    try:
        output = int(value.split('_')[-1])
        return output
    except:
        return 0


def get_relevant_keys(filename, groupname):
    vals = get_groups_of_h5(filename)
    current_vals = [split_and_convert_to_int(x) for x in vals if groupname in x]
    relevant_keys = [groupname + '_' + str(x) for x in current_vals if x > 0]
    return relevant_keys


def save_dataset_to_h5(filename, groupname, data):
    with h5py.File(filename, 'a') as h5f:
        h5f.create_dataset(groupname, data=data)


def load_dataset_from_h5(filename, groupname, idx=-1):
    with h5py.File(filename, "r") as h5f:
        if idx >= 0:
            data = h5f[groupname][idx]
        else:
            data = h5f[groupname][()]
        return data


def save_in_h5(filename, groupname, df_, is_continuous=False):
    if isfile(filename) and is_continuous:
        vals = get_groups_of_h5(filename)
        current_vals = [split_and_convert_to_int(x) for x in vals if groupname in x]
        if current_vals:
            last_number = max(current_vals)
            save_groupname = groupname + '_' + str(last_number + 1)
        else:
            save_groupname = groupname + '_' + str(1)
    else:
        save_groupname = groupname

    df_.to_hdf(filename, save_groupname)
    print(f'Saved in key {save_groupname} at file {filename}')



###########################################################
# Apply function with filter
def apply_function_with_filter(target_function, indexes, values, flags, group):
    if flags:
        [target_function(x, group, y) for (x, y, f) in zip(indexes, values, flags) if f]
    else:
        [target_function(x, group, y) for (x, y) in zip(indexes, values)]


def apply_function_with_filter_for_tensor(target_function, indexes, values, flags, group):
    [target_function(x, group, y.item()) for (x, y, f) in zip(indexes, values, flags) if f]


###########################################################
# Array manipulation
def pad_values(values, fixed_length, default_value):
    n = len(values)
    if n >= fixed_length:
        values_ = values[0:fixed_length]
    else:
        values_ = [default_value] * fixed_length
        values_[0:n] = values
    return values_

##########################################################
