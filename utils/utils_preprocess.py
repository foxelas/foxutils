# Import / Install libraries
import csv
import tarfile
import warnings
from datetime import datetime, timedelta
from os import listdir
from os.path import join as pathjoin

import numpy as np
import pandas as pd
import requests

from utils import utils

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)  # action='default' for testing


#####################################################


def get_start_and_end_dates(dataset_name):
    if dataset_name.lower() == "0_general_news":
        _start_date = datetime(2020, 8, 2)
        _end_date = datetime(2020, 11, 13) + timedelta(days=1)

    elif dataset_name.lower() == "1_newsapiorg":
        _start_date = datetime(2023, 1, 1)
        _end_date = datetime(2023, 2, 17)

    elif dataset_name.lower() == "2_nlpnews":
        _start_date = datetime(2015, 12, 2)
        _end_date = datetime(2023, 2, 2)

    else:
        raise NotImplementedError("This dataset does not exist.")

    return _start_date, _end_date


def get_target_dataset():
    return utils.target_dataset


start_date, end_date = get_start_and_end_dates(get_target_dataset())
lookback_window = utils.lookback_window
influence_hours = utils.influence_hours


def is_lookback_target(x, timestamp, lookback_hours=None):
    if not lookback_hours:
        lookback_hours = lookback_window
    flag = (x >= (timestamp - timedelta(hours=lookback_hours))) & (x < timestamp)
    return flag


def remove_none_and_pad_values(values, fixed_length=None, default_value=None):
    keep_indexes = [x is not None for x in values]
    if fixed_length is not None:
        values = [utils.pad_values(x, fixed_length, default_value) for x in values if x is not None]
    else:
        values = [x for x in values if x is not None]

    padded_values = np.array(values, dtype=float)
    return padded_values, keep_indexes


def get_historical_values_in_lookback_target(timestamp, timestamp_column, target_column, lookback_hours=None):
    historical_values = target_column[is_lookback_target(timestamp_column, timestamp, lookback_hours)]
    historical_values = np.array([x for x in historical_values], dtype=float)
    if historical_values.shape[0] == 0:
        return None
    return historical_values


def get_historical_values_in_lookback_target_2(timestamp, timestamp_column, target_column, lookback_hours=None):
    historical_values = target_column[is_lookback_target(timestamp_column, timestamp, lookback_hours)]
    historical_values = [x for x in historical_values]
    return historical_values


def get_average_historical_values_in_lookback_target(timestamp, timestamp_column, target_column, lookback_hours=None):
    historical_values = get_historical_values_in_lookback_target(timestamp, timestamp_column, target_column,
                                                                 lookback_hours)
    if historical_values is not None:
        historical_values_average = np.mean(historical_values, axis=0)
        return historical_values_average
    else:
        return None


def get_average_historical_values(timestamp_column, target_column, lookback_hours=None):
    if timestamp_column is None:
        values = [np.mean(x, axis=0) for x in target_column]
    else:
        values = [get_average_historical_values_in_lookback_target(x, timestamp_column, target_column, lookback_hours)
                  for x in timestamp_column]
    historical_values, keep_indexes = remove_none_and_pad_values(values)
    return historical_values, keep_indexes


def get_historical_values(timestamp_column, target_column, fixed_length, default_value, lookback_hours=None):
    if timestamp_column is None:
        values = [x for x in target_column]
    else:
        values = [get_historical_values_in_lookback_target(x, timestamp_column, target_column, lookback_hours) for x in
                  timestamp_column]
    historical_values, keep_indexes = remove_none_and_pad_values(values, fixed_length, default_value)
    return historical_values, keep_indexes


def get_future_currency_rate_and_trend(df_asset_price):
    future_close_values = [df_asset_price.loc[x + timedelta(hours=influence_hours)]['close'] for x in
                           df_asset_price.index if (x + timedelta(hours=influence_hours)) in df_asset_price.index]
    close_values = [df_asset_price.loc[x]['close'] for x in df_asset_price.index if
                    (x + timedelta(hours=influence_hours)) in df_asset_price.index]
    open_values = [df_asset_price.loc[x]['open'] for x in df_asset_price.index if
                   (x + timedelta(hours=influence_hours)) in df_asset_price.index]
    timestamp_values = [x.replace(tzinfo=None) for x in df_asset_price.index if
                        (x + timedelta(hours=influence_hours)) in df_asset_price.index]
    rate_values = [get_rate(x, y) for (x, y) in zip(open_values, future_close_values)]
    trend_values = [1 if x >= 0 else -1 for x in rate_values]
    trend_data = pd.DataFrame({'id': np.arange(0, len(timestamp_values), dtype=int), 'timestamp': timestamp_values,
                               'direction': trend_values, 'future_close': future_close_values,
                               'current_open': open_values, 'current_close': close_values, 'rate': rate_values})
    trend_data = trend_data.set_index('id')
    trend_data.sort_values(by=['timestamp'], ascending=True, inplace=True)
    return trend_data


def check_currency_trend(df_asset_price):
    print("Positive direction counts: " + str(df_asset_price['direction'].value_counts()[1]))
    print("Negative direction counts: " + str(df_asset_price['direction'].value_counts()[-1]))


def has_all_required_info(article_info):
    if article_info['title'] and article_info['content'] and article_info['language'] == 'English':
        return True
    return False


#####################################################
# Read news data

def read_info_from_news_item(sequence_id):
    filename = pathjoin(utils.datasets_dir, utils.target_dataset, utils.extracted_folder,
                        str(sequence_id) + '.json.gz')
    json_data = utils.load_data_from_json_gz(filename)

    article_title = utils.get_item_from_json_file(json_data, 'title')
    article_content = utils.get_item_from_json_file(json_data, 'content')
    article_timestamp = utils.get_item_from_json_file(json_data, 'publishedDate')
    article_estimated_timestamp = utils.get_item_from_json_file(json_data, 'estimatedPublishedDate')
    article_breaking = 1.0 if 'breaking' in article_title.lower() else 0.0

    article_wordcount = utils.get_item_from_json_file(json_data, 'wordCount')

    sentiment_dict = utils.get_item_from_json_file(json_data, 'sentiment')
    sentiment = float(sentiment_dict['score']) if sentiment_dict else 0.0

    article_info = {'sequence_id': int(sequence_id),
                    'timestamp': article_timestamp,
                    'estimated_timestamp': article_estimated_timestamp,
                    'id': int(utils.get_item_from_json_file(json_data, 'id')),
                    'language': utils.get_item_from_json_file(json_data, 'language'),
                    'sentiment': sentiment,
                    'content': article_content,
                    'title': article_title,
                    'word_count': int(article_wordcount),
                    'is_breaking': article_breaking
                    }
    return article_info


def read_value_from_news_item_with_key(sequence_id, key):
    filename = pathjoin(utils.datasets_dir, utils.target_dataset, utils.extracted_folder,
                        str(sequence_id) + '.json.gz')
    json_data = utils.load_data_from_json_gz(filename)

    return utils.get_item_from_json_file(json_data, key)


def load_datasets_as_numpy(target_ids, preparation_batch_size, key):
    return np.array([load_from_preprocessed_data(aa_id, preparation_batch_size, key) for aa_id in target_ids])


#####################################################

def tar_all_files_per_month(dataset_dir, previous_year, previous_month, delimeter):
    # compress all extracted files from previous month to a tar file
    target_files = listdir(dataset_dir)
    target_files = [i for i in target_files if ('.json.gz' in i)]
    save_tar_filename = pathjoin(dataset_dir, utils.settings['DIRECTORY']['extracted_folder'],
                                 previous_year + delimeter + previous_month + '.tar')
    utils.compress_files_to_tar(save_tar_filename, target_files, dataset_dir, '.json.gz')
    print(f'Saved data for {previous_year}-{previous_month} in {save_tar_filename}.')


def extract_news_files(input_dir=None):
    dataset_dir = pathjoin(utils.datasets_dir, utils.target_dataset, utils.settings['DIRECTORY']['extracted_folder'])
    print(f"Extracting .tar files from {input_dir}")

    utils.extract_files_from_tar(input_dir, dataset_dir)
    print(f"Saved files in {dataset_dir}")


def prepare_news_files():
    # Extracts JSON files from .tar and decouples them so each JSON file contains one article
    dataset_dir = pathjoin(utils.datasets_dir, utils.target_dataset)
    print(f"Reading files from {dataset_dir}")

    data_files = listdir(dataset_dir)
    data_files = [i for i in data_files if ('.tar' in i) and not ('lnl' in i)]
    previous_month = '12'
    previous_year = '2015'
    delimeter = '-'

    for tar_file in data_files:
        filename = pathjoin(dataset_dir, tar_file)
        tar = tarfile.open(filename)
        json_files = [i.name for i in tar.getmembers() if "json.gz" in i.name]
        current_month = tar_file.split(delimeter)[1]

        if not (current_month == previous_month):
            tar_all_files_per_month(dataset_dir, previous_year, previous_month, delimeter)
            previous_year = tar_file.split(delimeter)[0]
            previous_month = current_month

        if json_files:
            for target_file in json_files:
                json_data = utils.load_data_from_json_gz(tar.extractfile(target_file))
                if json_data['status'] == 'SUCCESS' and len(json_data['articles']) > 0:
                    for target_article in json_data['articles']:
                        sequence_id = target_article['sequenceId']
                        save_filename = pathjoin(dataset_dir, sequence_id + '.json.gz')
                        utils.save_data_to_json_gz(save_filename, target_article)

    # for the last month: compress all extracted files from previous month to a tar file
    tar_all_files_per_month(dataset_dir, previous_year, previous_month, delimeter)


#########################################################################
# Asset data

def get_kaiko_data_single_index(instrument_pair_name='btc-usd', interval='1h', page_size='10000'):
    headers = {
        'Accept': 'application/json',
        'X-Api-Key': utils.get_api_key("kaiko.json")
    }

    url = 'https://us.market-api.kaiko.io/v2/data/trades.v1/exchanges/cbse/spot/' + instrument_pair_name \
          + '/aggregations/ohlcv?interval=' + interval + '&page_size=' + page_size
    response = requests.get(url, headers=headers, stream=True, verify=False)
    data = response.json()

    total_pages = 0
    data_list = []
    if data['result'] == 'success':
        total_pages += 1
        data_list.append(data['data'])
        while 'next_url' in data.keys():
            total_pages += 1
            print(f'Total pages {total_pages}')
            new_url = data['next_url']
            response = requests.get(new_url, headers=headers, stream=True, verify=False)
            data = response.json()

            if data['result'] == 'success':
                data_list.append(data['data'])

        flat_list = [item for sublist in data_list for item in sublist]
        return flat_list

    return data_list


def save_kaiko_data_single_index_data_frame(interval='1h'):
    data_list = get_kaiko_data_single_index(instrument_pair_name='btc-usd', interval=interval)
    df = pd.DataFrame(data_list)
    df['datetime'] = [utils.get_datetime_from_unix_timestamp(x) for x in df['timestamp']]
    df = df[df['datetime'] > (start_date - timedelta(days=1))]
    df.dropna(inplace=True)
    df['open'] = pd.to_numeric(df['open'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['close'] = pd.to_numeric(df['close'])
    df['volume'] = pd.to_numeric(df['volume'])
    save_name = utils.asset_group + '_raw'
    if interval != '1h':
        save_name = save_name + '_' + interval
    utils.save_in_h5(get_asset_data_filename(), save_name, df)


def get_asset_data_filename():
    return pathjoin(utils.datasets_dir, utils.settings['DIRECTORY']['btc_data_folder'],
                    utils.settings['BTC']['target_btc_file'])


def prepare_raw_asset_price_files(interval='1h'):
    filename = get_asset_data_filename()
    if '.h5' in filename:
        save_name = utils.asset_group + '_raw'
        if interval != '1h':
            save_name = save_name + '_' + interval
        df_asset_price = pd.read_hdf(filename, save_name)
    else:
        df_asset_price = pd.read_csv(filename)
        df_asset_price.datetime = pd.to_datetime(df_asset_price.datetime)

    df_asset_price.columns = df_asset_price.columns.str.lower()
    df_asset_price.sort_values(by='datetime', ascending=True, inplace=True)
    df_asset_price.reset_index(drop=True, inplace=True)
    df_asset_price.set_index(['datetime'], inplace=True)
    df_asset_price = df_asset_price[~df_asset_price.index.duplicated(keep='first')]

    print(df_asset_price.tail())
    btc_data_dir = filename.replace('.csv', '.h5')
    df_asset_price.to_hdf(btc_data_dir, utils.asset_group)
    print(f'Saved in key {utils.asset_group} at file {btc_data_dir}')


def get_rate(open_value, close_value):
    rate_value = (close_value - open_value) / np.abs(open_value)
    return rate_value


def get_change_rate(current_value, next_value):
    return get_rate(current_value, next_value)


def preprocess_asset_price_files():
    df_asset_price = get_asset_data(is_preprocessed=False)
    print('Before')
    print(min(df_asset_price.index))
    print(max(df_asset_price.index))

    target_ids = [x for x in df_asset_price.index if (x + timedelta(hours=influence_hours)) in df_asset_price.index]
    future_open_values = [df_asset_price.loc[x + timedelta(hours=influence_hours)]['open'] for x in target_ids]
    open_values = [df_asset_price.loc[x]['open'] for x in target_ids]
    close_values = [df_asset_price.loc[x]['close'] for x in target_ids]
    volume_values = [df_asset_price.loc[x]['volume'] for x in target_ids]
    high_values = [df_asset_price.loc[x]['high'] for x in target_ids]
    low_values = [df_asset_price.loc[x]['low'] for x in target_ids]

    timestamp_values = [x.replace(tzinfo=None) for x in target_ids]
    rate_values = [get_change_rate(x, y) for (x, y) in zip(open_values, future_open_values)]
    trend_values = [1 if x >= 0 else -1 for x in rate_values]
    trend_data = pd.DataFrame({'id': np.arange(0, len(timestamp_values), dtype=int), 'timestamp': timestamp_values,
                               'direction': trend_values, 'future_open': future_open_values,
                               'current_open': open_values, 'rate': rate_values, 'close': close_values,
                               'volume': volume_values, 'high': high_values, 'low': low_values})
    trend_data = trend_data.set_index('id')

    trend_data.sort_values(by=['timestamp'], ascending=True, inplace=True)

    print('After')
    print(min(trend_data['timestamp']))
    print(max(trend_data['timestamp']))

    btc_data_dir = get_asset_data_filename().replace('.csv', '.h5')
    trend_data.to_hdf(btc_data_dir, utils.preprocessed_asset_group)
    print(f'Saved in key {utils.preprocessed_asset_group} at file {btc_data_dir}')
    return trend_data


def get_asset_data(is_preprocessed=False):
    filename = get_asset_data_filename().replace('.csv', '.h5')
    if is_preprocessed:
        df_asset_price = pd.read_hdf(filename, utils.preprocessed_asset_group)
    else:
        df_asset_price = pd.read_hdf(filename, utils.asset_group)
    return df_asset_price


################################################################
# Prepare news items grouped by hour
def write_target_filenames(extension='json.gz'):
    dataset_dir = pathjoin(utils.datasets_dir, utils.target_dataset, utils.extracted_folder)
    print(f"Loading files from {dataset_dir}")

    data_files = listdir(dataset_dir)
    sequence_ids = [i.split('.')[0] for i in data_files if (extension in i)]

    filename = get_filename_in_preprocessed_folder('sequence_ids', '.csv')
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerows(zip(list(range(len(sequence_ids))), sequence_ids))

    print(f'Saved target filenames to: {filename}')


def read_target_filenames():
    filename = get_filename_in_preprocessed_folder('sequence_ids', '.csv')
    sequence_ids_df = pd.read_csv(filename, header=None, index_col=None)
    print(f'Read {sequence_ids_df.shape[0]} target filenames from: {filename}')
    return sequence_ids_df[1].values.tolist()


def read_check_and_return_info_from_news_item(sequence_id):
    article_info_list = read_info_from_news_item(sequence_id)
    is_complete_data = has_all_required_info(article_info_list)
    estimated_timestamp = pd.to_datetime(article_info_list['estimated_timestamp'])
    word_count = article_info_list['word_count']
    is_breaking = article_info_list['is_breaking']
    sentiment = article_info_list['sentiment']
    return dict(sequence_id=sequence_id, timestamp=estimated_timestamp, word_count=word_count, is_breaking=is_breaking,
                sentiment=sentiment, is_complete_data=is_complete_data)


def write_target_filenames_grouped():
    sequence_ids = read_target_filenames()
    data_items = [read_check_and_return_info_from_news_item(sequence_id) for sequence_id in sequence_ids]
    df = pd.DataFrame(data_items)

    df_hourly = pd.DataFrame()
    grouping = [df['timestamp'].dt.year, df['timestamp'].dt.month, df['timestamp'].dt.day, df['timestamp'].dt.hour]
    temp_grouping = df.groupby(grouping)

    df_hourly['target_ids'] = temp_grouping.apply(lambda x: np.array(x['sequence_id']))
    df_hourly['aa_ids'] = temp_grouping.apply(lambda x: np.array(x.index))
    # df_hourly['target_ids'] = temp_grouping.apply(lambda vals: np.array([x['sequence_id'] for x in vals if x['is_complete_data']]))
    # df_hourly['aa_ids'] = temp_grouping.apply(lambda vals: np.array([x.index for x in vals if x['is_complete_data']]))
    datetimes = [x for x in temp_grouping.timestamp.min()]
    datetimes = [datetime(x.year, x.month, x.day, x.hour) for x in datetimes]
    df_hourly['hourly_timestamp'] = datetimes
    df_hourly['average_word_count'] = temp_grouping.word_count.mean()
    df_hourly['num_published'] = temp_grouping.sequence_id.count()
    df_hourly['average_sentiment'] = temp_grouping.sentiment.mean()
    df_hourly['average_is_breaking'] = temp_grouping.is_breaking.mean()
    df_hourly['is_complete_data'] = temp_grouping.is_complete_id.any()
    df_hourly.reset_index(drop=True, inplace=True)

    filename = get_filename_in_preprocessed_folder(utils.dataset_filename, '.h5')
    save_key = 'serially'
    utils.save_in_h5(filename, save_key, df)
    print(f'Saved serially arranged SequenceIds for news items in key {save_key} at file {filename}')

    save_key = 'hourly'
    utils.save_in_h5(filename, save_key, df_hourly)
    print(f'Saved hourly-grouped SequenceIds for news items in key {save_key} at file {filename}')


def read_target_filenames_grouped():
    filename = get_filename_in_preprocessed_folder(utils.dataset_filename, '.h5')
    return pd.read_hdf(filename, 'hourly')


def filename_for_saving():
    return 'preprocessed_data'


################################################################
# Prepare features

def get_filename_in_preprocessed_folder(filename, extension='.h5'):
    filename = str(filename)
    if not (extension in filename):
        filename = filename + extension
    return pathjoin(utils.datasets_dir, utils.target_dataset, utils.preprocessed_folder, filename)


def load_from_preprocessed_data(aa_id, preparation_batch_size, key):
    batch_number = aa_id // preparation_batch_size
    id_in_batch = aa_id % preparation_batch_size
    filename = get_filename_in_preprocessed_folder(filename_for_saving())
    value = utils.load_dataset_from_h5(filename, utils.join_h5_keys([key, str(batch_number)]))[id_in_batch] #\
        #.decode(utils.encoding)
    return value


def save_to_preprocessed_data(batch_number, keys, values):
    filename = get_filename_in_preprocessed_folder(filename_for_saving())
    for (key, value) in zip(keys, values):
        utils.save_dataset_to_h5(filename, utils.join_h5_keys([key, str(batch_number)]), value)


def read_info_from_news_item_by_key(sequence_id, key):
    return read_info_from_news_item(sequence_id)[key]

