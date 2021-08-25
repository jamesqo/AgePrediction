import glob
import random

import numpy as np
from sklearn.model_selection import train_test_split
import torch

from .train import SPLITS_DIR, load_samples, resample

def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    split_fnames = glob.glob(f"{SPLITS_DIR}/nfold_imglist_all_nfold_*.list")
    assert len(split_fnames) == 5

    '''
    df = load_samples(split_fnames, bin_width=1, min_bin_count=1, max_samples=None)
    print(df.shape[0])
    print(np.mean(df['age']))
    print(np.std(df['age']))
    print(sum((df['sex'] == 'M') | (df['sex'] == 'm')) + 1)
    print(sum((df['sex'] == 'F') | (df['sex'] == 'f')))
    print(df['sex'].value_counts())
    '''

    df = load_samples(split_fnames, bin_width=1, min_bin_count=10, max_samples=None)
    _train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['agebin'])
    for mode in ('none', 'over', 'under'):
        train_df = resample(_train_df, mode=mode, oversamp_limit=126)
        print("========")
        print(df.shape[0])
        print(train_df.shape[0])
        print(val_df.shape[0])
        print("========")

if __name__ == '__main__':
    main()
