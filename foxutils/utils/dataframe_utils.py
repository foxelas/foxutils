import pandas as pd

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