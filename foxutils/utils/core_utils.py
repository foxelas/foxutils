import configparser
import gzip
import json
import os
import tarfile
from datetime import datetime
from os import listdir, remove, makedirs
from os.path import exists as pathexists
from os.path import join as pathjoin
from os.path import normpath, isfile, dirname, split
from sys import getsizeof
import requests
from pathlib import Path
import re
import glob
from os import getcwd, sep

import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import torch
import time

from PIL import Image, ImageFile
from urllib3.exceptions import MaxRetryError, NewConnectionError

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms

###########################################################

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['IF_CUDNN_DETERMINISTIC'] = '1'  # for tf 2.0+
random.seed(SEED)
np.random.seed(SEED)
tf.random.seed(SEED)

###########################################################
# The filename of the settings file
settings_filename = 'config.ini'


# Reads settings from file
def read_config():
    config = configparser.ConfigParser()
    c = 0
    filepath = ''
    while not pathexists(pathjoin(filepath, settings_filename)) and c < 20:
        filepath = pathjoin('../..', filepath)
        c = c + 1

    if c == 20:
        raise FileNotFoundError('Missing config.ini file!')

    filepath = pathjoin(filepath, settings_filename)
    config.read(filepath)
    # print({section: dict(config[section]) for section in config.sections()})
    return config


settings = read_config()

datasets_dir = normpath(settings['DIRECTORY']['datasets_dir'])
models_dir = normpath(settings['DIRECTORY']['models_dir'])
token_dir = normpath(settings['DIRECTORY']['token_dir'])
preprocessed_folder = settings['DIRECTORY']['preprocessed_folder']
extracted_folder = settings['DIRECTORY']['extracted_folder']

is_test = settings['RUN']['is_test']
test_suffix = '_test' if is_test else ''
project_name = settings['RUN']['project_name']

encoding = "utf-8"

###########################################################
base_folder = 'github'


def get_package_path(name='foxutils'):
    target_path = get_base_path()
    target_path = target_path.split(sep)
    target_path.insert(len(target_path), name)
    target_path.insert(len(target_path), '')
    target_path.insert(1, sep)
    target_path = pathjoin(*target_path)
    # target_path = pathjoin('..', '..', '..', 'foxutils', '')
    return target_path


def get_base_path():
    cwd = getcwd()  # *\github\EMIA
    target_path = normpath(cwd)
    target_path = target_path.split(sep)
    target_path = target_path[0:target_path.index(base_folder) + 1]
    target_path.insert(1, sep)
    target_path = pathjoin(*target_path)
    return target_path


print(f'Default package path is {get_package_path()}')


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
    except ValueError:
        try:
            date_value = datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S%z")
        except ValueError:
            try:
                date_value = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                date_value = datetime.strptime(date_string, "%Y-%m-%d %I-%p")

    return date_value


def convert_fully_connected_string_to_datetime(date_string):
    try:
        date_value = datetime.strptime(date_string, "%Y%m%d%H%M%S")

    except ValueError:
        return None

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


def increment_path(path, exist_ok=True, sep=''):
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


###########################################################
# Memory

def get_device():
    device_ = "cuda" if torch.cuda.is_available() else "cpu"
    if __name__ == '__main__':
        print(f'Running on {device_}')
    return device_


device = get_device()


def show_gpu_settings():
    print("Torch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("GPU availability:", torch.cuda.is_available())
    print("Number of GPU devices:", torch.cuda.device_count())
    print("Name of current GPU:", torch.cuda.get_device_name(0))


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
# IMG files

def get_request(link, **kwargs):
    success = False
    r = None
    try:
        r = requests.get(link, **kwargs)
        if r.status_code == 403:
            print(f'Access denied for link {link}')
        else:
            success = True

    except requests.exceptions.ConnectionError as ce:
        print(f'A connection error occurred for {link}.')
        print(ce)

    except (MaxRetryError, NewConnectionError,
            requests.exceptions.NewConnectionError, requests.exceptions.SSLError) as ne:
        print(f'A request error occurred for {link}.')
        print(ne)

    return success, r


def save_image_from_link(image_url, image_name='img.jpg'):
    success, r = get_request(image_url)
    if success:
        img_data = r.content
        with open(image_name, 'wb') as handler:
            handler.write(img_data)


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

def merge_data_frames(dfs, method='outer', use_interpolation=None, index_column='datetime', dropna=True):
    dfs = [x.set_index([index_column]) for x in dfs]
    df = pd.concat(dfs, join=method)
    df.sort_index(inplace=True)

    if use_interpolation is not None:
        for i, x in enumerate(use_interpolation):
            if x is not None:
                interp_cols = dfs[i].columns.tolist()
                [df[col].interpolate(method=x, inplace=True) for col in interp_cols]

    if dropna:
        df.dropna(inplace=True)

    return df


##########################################################

def time_execution(target_function, **kwargs):
    start = time.time()
    target_function(**kwargs)
    end = time.time()
    print(f'Execution took {end - start} seconds.')


#########################################################

def mkdir_if_not_exist(target_path):
    target_path = dirname(target_path)
    if not pathexists(target_path):
        makedirs(target_path)


#########################################################

def flatten(l):
    return [item for sublist in l for item in sublist]


#########################################################

def read_image_to_tensor(filename, dataset_dir, im_height=None, im_width=None):
    image = Image.open(pathjoin(dataset_dir, filename))
    if im_width is not None and im_height is not None:
        image = image.resize((im_height, im_width))
    image = transforms.PILToTensor()(image)
    image = image.float()
    image /= 255.0
    return image
