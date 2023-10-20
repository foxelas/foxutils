import configparser
import glob
import gzip
import json
import re
import tarfile
import time
from datetime import datetime
from os import getcwd, sep
from os import listdir, remove, makedirs
from os.path import exists as pathexists
from os.path import join as pathjoin
from os.path import normpath, dirname
from pathlib import Path

import requests
import torch
from urllib3.exceptions import MaxRetryError, NewConnectionError

###########################################################

SEED = 42

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


def get_device():
    if 'device' in settings['RUN'].keys():
        device_ = settings['RUN']['device']
    else:
        device_ = "cuda" if torch.cuda.is_available() else "cpu"

    if __name__ == '__main__':
        print(f'Running on {device_}')
    return device_


device = get_device()

datasets_dir = normpath(settings['DIRECTORY']['datasets_dir'])
models_dir = normpath(settings['DIRECTORY']['models_dir'])
token_dir = normpath(settings['DIRECTORY']['token_dir'])

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


# print(f'Default package path is {get_package_path()}')


###########################################################
# Tokens for API access
def get_api_key(filename):
    token_data = json.load(open(pathjoin(token_dir, filename)))
    token = token_data.get("key")
    return token


###########################################################
# Datetimes

def get_current_datetime(tz=None):
    current_datetime = datetime.now()
    if tz is not None:
        current_datetime = current_datetime.astimezone(tz)

    return current_datetime


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


def find_files_by_extension(filepath, target_extension, ascending=True):
    file_list = listdir(filepath)
    file_list = [x for x in file_list if target_extension in x]
    file_list.sort(reverse=not ascending)
    return file_list

#########################################################

def flatten(l):
    return [item for sublist in l for item in sublist]

#########################################################
import logging

def set_logger(logger):
    logging_level = settings['RUN']['logging']
    if logging_level == 'INFO':
        logger.setLevel(logging.INFO)
    elif logging_level == 'DEBUG':
        logger.setLevel(logging.DEBUG)
    return logger

#########################################################