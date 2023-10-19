from os.path import isfile

import h5py


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

