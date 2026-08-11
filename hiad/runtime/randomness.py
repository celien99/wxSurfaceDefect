import os
import random

import numpy as np
import torch


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic convolution selection makes fixed-shape Dinomaly training
    # needlessly slow. Random sources remain seeded for repeatable sampling.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
