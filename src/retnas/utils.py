import yaml
import shutil
import random
import numpy as np
import torch

def print_banner():
    print("RETNAS — Real Time Neural Adaptive Scheduling")

class Config:

    def __init__(self, **entries):
        self.__dict__.update(entries)

def dump_yaml(file, data):
    with open(file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def load_yaml(file):
    with open(file, 'r') as f:
        return yaml.safe_load(f)

def copyfile(src, dst):
    shutil.copy(src, dst)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
