import pandas as pd
import numpy as np
from datetime import timedelta
##########################################################

def merge_data_frames(dfs, method='outer', use_interpolation=None, index_column='datetime', dropna=True):
    dfs = [x.set_index([index_column]) for x in dfs]
    df = pd.concat(dfs, join=method)
    df.sort_index(inplace=True)

    if use_interpolation is not None:
        for i, x in enumerate(use_interpolation):
            if x is not None:
                interp_cols = dfs[i].columns.tolist()
                if x == 'nearest':
                    [df[col].interpolate(method=x, fill_value='extrapolate', inplace=True) for col in interp_cols]
                else:
                    [df[col].interpolate(method=x, inplace=True) for col in interp_cols]
    if dropna:
        df.dropna(inplace=True)

    if len(df) == 0:
        print(
            "The requested dataframe is empty! Check if you have used Categorical values, might need to encode them.\n")

    return df


def check_for_missing_dates(df, mode="fixed_intervals", interval_limit=None):
    if 'datetime' in df.columns:
        timestamps = df['datetime'].values
    else:
        timestamps = df.index.values

    diffs = np.diff(timestamps)
    if mode == "fixed_intervals":
        un_diffs, un_counts = np.unique(diffs, return_counts=True)
        un_diffs = np.delete(un_diffs, np.argmax(un_counts))

    if mode == "consecutive_intervals":
        if not interval_limit:
            interval_limit = np.timedelta64(15, 'm')

        un_diffs = [x for x in diffs if x >= interval_limit]

    missing_ids = [i for i, val in enumerate(diffs) if val in un_diffs]
    for i in missing_ids:
        interval = pd.to_timedelta(diffs[i])
        print(f"Missing values between {timestamps[i]} and {timestamps[i + 1]}. Timedelta: {interval}.")

    return missing_ids

def split_data_frame_at_missing_data(df, mode=None, missing_ids=None, interval_limit=None):
    if missing_ids is None:
        missing_ids = check_for_missing_dates(df, mode, interval_limit)

    class_df_dict = {elem: pd.DataFrame() for elem in range(len(missing_ids))}

    # split dataframes with non-consecutive data
    start_id = 0
    for key in class_df_dict.keys():
        end_id = missing_ids[key] + 1
        #print(f'start {start_id} end {end_id}')
        class_df_dict[key] = df.iloc[start_id:end_id]
        start_id = end_id

    class_df_dict[key+1] = df.iloc[start_id:]

    #print(f'Sum of dfs {np.array([len(class_df_dict[x]) for x in class_df_dict.keys()]).sum()}')
    #print(f'Original df {len(df)}')
    return class_df_dict, missing_ids


def encode_categorical_values(df, target_column):
    categ_dict = {x: i for i, x in enumerate(np.unique(df[target_column]))}
    inv_categ_dict = {v: k for k, v in categ_dict.items()}
    df[target_column] = df[target_column].apply(lambda x: categ_dict[x])
    return categ_dict, inv_categ_dict, df

