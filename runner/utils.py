import json
import numpy as np
import os
import pandas as pd


def get_seed(save_path, load_seed):
    if load_seed and os.path.exists('{}/setting.json'.format(save_path)):
        with open('{}/setting.json'.format(save_path), mode='rt') as f:
            setting = json.load(f)
            return setting['seed']
    else:
        return np.random.randint(0, 2 ** 32)


def save_setting(save_path, **params):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    with open('{}/setting.json'.format(save_path), mode='wt') as f:
        json.dump(params, f)


def save_result(save_path, result, log=None):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    with open('{}/result.json'.format(save_path), mode='wt') as f:
        json.dump(result, f)
    if log is not None:
        df = pd.DataFrame(log)
        df.index.name = '#index'
        df.to_csv('{}/results.csv'.format(save_path))