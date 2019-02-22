import numpy as np
import pandas as pd
import csv


def load_part(part, path='gapping/'):
    if part == 'train':
        return pd.read_csv(path+'train.csv', sep='\t', quoting=csv.QUOTE_NONE).fillna('')
    elif part == 'dev' or part == 'valid':
        return pd.read_csv(path+'dev.csv', sep='\t', quoting=csv.QUOTE_NONE).fillna('')
    elif part == 'test':
        return pd.read_csv(path+'test.csv', sep='\t', quoting=csv.QUOTE_NONE).fillna('')
    else:
        raise NotImplementedError


def load_df(parts=('train', 'dev', 'test'), path='gapping/'):
    res = {}
    for part in parts:
        res[part] = load_part(part=part, path=path)
    return res
