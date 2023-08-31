import pandas as pd
import numpy as np
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

def check_for_missing_dates(df):
    timestamps = df['datetime']
    diffs = np.diff(timestamps)
    un_diffs, un_counts = np.unique(diffs, return_counts=True)
    un_diffs = np.delete(un_diffs, np.argmax(un_counts))
    for val in un_diffs:
        idx = np.where(diffs == val)[0]
        for i in idx:
            print(f"Missing values between {df.iloc[i]['datetime']} and {df.iloc[i + 1]['datetime']} ")

    print("")
