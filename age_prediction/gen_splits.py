"""
Split the original dataframe into fixed training and test sets.
Provide under/oversampled versions of the training set.
"""

import argparse
import re
import os
import glob

import nibabel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
FOLDS_DIR = os.path.join(ROOT_DIR, "folderlist")

def load_samples(fold_fnames, min_bin_count=10, max_samples=None):
    """
    Loads a dataframe from the fold filenames.

    min_bin_count: Minimum number of samples an age bin must have to be included
    max_samples: Maximum number of samples allowed in the resulting dataframe (used for debugging purposes)
    """
    def correct_path(path):
        path = path.replace('/ABIDE/', '/ABIDE_I/')
        path = path.replace('/NIH-PD/', '/NIH_PD/')
        return path
    
    def extract_dataset(path):
        name = re.match(r'/neuro/labs/grantlab/research/MRI_Predict_Age/([^/]*)', path)[1]
        return 'MGHBCH' if name in ('MGH', 'BCH') else name
    
    def get_pkl_path(row):
        img_path, dataset, id_ = row['img_path'], row['dataset'], row['id']
        out_path = os.path.join(ROOT_DIR, "pickles", dataset, f"{id_}.npy")
        if not os.path.isfile(out_path):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            image = nibabel.load(img_path).get_fdata()
            image = image[54:184, 25:195, 12:132] # Crop out zeroes
            image /= np.percentile(image, 95) # Normalize intensity
            np.save(out_path, image)
        return out_path

    schema = {'id': str, 'age': float, 'sex': str, 'img_path': str}
    dfs = []
    for fname in fold_fnames:
        fold_num = int(fname[:-len('.list')].split('_')[-1])

        df = pd.read_csv(fname, sep=' ', header=None, names=['id', 'age', 'sex', 'img_path'], dtype=schema)
        df['img_path'] = df['img_path'].apply(correct_path)
        df['agebin'] = df['age'] // 1
        df['fold'] = fold_num
        df['dataset'] = df['img_path'].apply(extract_dataset)
        df['pkl_path'] = df.apply(get_pkl_path, axis=1)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, axis=0)
    ids = combined_df['id']
    assert len(ids) == len(set(ids)) # Patient
    
    if max_samples is not None:
        combined_df = combined_df.sample(max_samples, random_state=42)

    if min_bin_count is not None:
        bin_counts = combined_df['agebin'].value_counts()
        bins_below_cutoff = [bin for bin in bin_counts.keys() if bin_counts[bin] < min_bin_count]
        combined_df = combined_df[~combined_df['agebin'].isin(bins_below_cutoff)]
    
    # Filter out files that don't exist
    exists = combined_df['img_path'].apply(os.path.isfile)
    missing_files = combined_df['img_path'][~exists]
    print(f"{len(missing_files)} file(s) are missing:")
    print('\n'.join(missing_files))
    combined_df = combined_df[exists]
    
    return combined_df

def resample(df, mode, deflation_limit, inflation_limit):
    bins = sorted(set(df['agebin']))
    bin_counts = {bin: sum(df['agebin'] == bin) for bin in bins}
    bin_ratios = {bin: count / df.shape[0] for bin, count in bin_counts.items()}

    if mode == 'over':
        for bin in bins:
            n_under = inflation_limit - bin_counts[bin]
            if n_under <= 0:
                # Undersample when bin count is over the limit
                new_samples = df[df['agebin'] == bin].sample(inflation_limit, replace=False, random_state=42)
                df = df[df['agebin'] != bin]
                df = pd.concat([df, new_samples], axis=0)
            else:
                # Oversample as usual
                new_samples = df[df['agebin'] == bin].sample(n_under, replace=True, random_state=42)
                df = pd.concat([df, new_samples], axis=0)
    elif mode == 'under':
        for bin in bins:
            n_over = bin_counts[bin] - deflation_limit
            if n_over <= 0:
                # Oversample when bin count is under the limit
                new_samples = df[df['agebin'] == bin].sample(-n_over, replace=True, random_state=42)
                df = pd.concat([df, new_samples], axis=0)
            else:
                # Undersample as usual
                new_samples = df[df['agebin'] == bin].sample(deflation_limit, replace=False, random_state=42)
                df = df[df['agebin'] != bin]
                df = pd.concat([df, new_samples], axis=0)
    elif mode == 'scale-up' or mode == 'scale-down':
        if mode == 'scale-up':
            target_count = inflation_limit * len(bins)
            replace = True
        else:
            target_count = deflation_limit * len(bins)
            replace = False

        for bin in bins:
            count = int(bin_ratios[bin] * target_count)
            new_samples = df[df['agebin'] == bin].sample(count, replace=replace, random_state=42)
            df = df[df['agebin'] != bin]
            df = pd.concat([df, new_samples], axis=0)
    else:
        raise Exception(f"Invalid sampling mode: {mode}")

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dirname', type=str, help='name of directory to save splits to')
    parser.add_argument('--val-size', type=float, default=0.2, help='size of validation set used for training')
    parser.add_argument('--min-bin-count', type=int, default=10, help='minimum number of samples per age bin (bins with fewer samples will be removed)')
    parser.add_argument('--deflation-limit', type=int, default=18, help='minimum number of samples each age bin will be deflated to when undersampling (bins with fewer samples will be oversampled)')
    parser.add_argument('--inflation-limit', type=int, default=126, help='maximum number of samples each age bin will be inflated to when oversampling (bins with more samples will be undersampled)')
    # For debugging purposes only
    parser.add_argument('--max-samples', type=int, help='limit the number of samples used for training/validation')
    args = parser.parse_args()

    out_dir = os.path.join(ROOT_DIR, "splits", args.dirname)
    os.makedirs(out_dir, exist_ok=True)

    fold_fnames = glob.glob(f"{FOLDS_DIR}/nfold_imglist_all_nfold_*.list")
    assert len(fold_fnames) == 5

    RESAMPLING_MODES = ['over', 'under', 'scale-up', 'scale-down'] if args.max_samples is None else []

    df = load_samples(fold_fnames, args.min_bin_count, args.max_samples)

    train_df, val_df = train_test_split(df, test_size=args.val_size, stratify=df['agebin'], random_state=42)
    train_df.to_csv(os.path.join(out_dir, "train"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val"), index=False)
    for mode in RESAMPLING_MODES:
        resampled_train_df = resample(train_df, mode, args.deflation_limit, args.inflation_limit)
        resampled_train_df.to_csv(os.path.join(out_dir, f"train_{mode}"), index=False)
    
    print("Done")

if __name__ == '__main__':
    main()
