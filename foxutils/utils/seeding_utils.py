import os
import numpy as np
import random
import tensorflow as tf
import torch

from .core_utils import SEED

def initialize(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['IF_CUDNN_DETERMINISTIC'] = '1'  # for tf 2.0+
    random.seed(seed)
    rng = np.random.default_rng(seed)
    tf.random.seed(seed)
    torch.seed(seed)