from sys import getsizeof

import torch
import pandas as pd

from .core_utils import device


###########################################################
# Memory

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
