from sys import getsizeof

import pandas as pd


###########################################################
# GPU

def show_gpu_settings_torch():
    try:
        import torch

        print("\nCheck Torch version:", torch.__version__)
        print("GPU availability:", torch.cuda.is_available())

        # setting device on GPU if available, else CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        print("CUDA version:", torch.version.cuda)

        # Additional Info when using cuda
        if device.type == "cuda":
            print("Number of GPU devices:", torch.cuda.device_count())
            print("Name of current GPU:", torch.cuda.get_device_name(0))
            print("Memory Usage:")
            print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
            print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), "GB")
    except ModuleNotFoundError:
        print("Torch is not installed.")

def show_gpu_settings_tensorflow():
    try:
        import tensorflow as tf
        print(f"\nCheck tensorflow version: {tf.__version__}")
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    except ModuleNotFoundError:
        print("Tensorflow is not installed.")


def test_gpu(values=["torch", "tensorflow"]):
    print("Testing GPU settings...")
    if "torch" in values:
        show_gpu_settings_torch()
        print("")

    if "tensorflow" in values:
        show_gpu_settings_tensorflow()
        print("")

###########################################################
# Memory
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
