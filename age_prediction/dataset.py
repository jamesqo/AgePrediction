import nibabel
import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
from torch.utils import data

from .lds import get_lds_kernel_window

# todo: does this need to be changed too?
def prepare_weights(df, reweight, lds, lds_kernel, lds_ks, lds_sigma):
    if reweight == 'none':
        return None
    
    bin_counts = df['agebin'].value_counts()
    if reweight == 'inv':
        num_per_label = [bin_counts[bin] for bin in df['agebin']]
    elif reweight == 'sqrt_inv':
        num_per_label = [np.sqrt(bin_counts[bin]) for bin in df['agebin']]
    
    if lds:
        lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        smoothed_value = pd.Series(
            convolve1d(bin_counts.values, weights=lds_kernel_window, mode='constant'),
            index=bin_counts.index)
        num_per_label = [smoothed_value[bin] for bin in df['agebin']]

    weights = [1. / x for x in num_per_label]
    scaling = len(weights) / np.sum(weights)
    weights = [scaling * x for x in weights]
    return weights

class AgePredictionDataset(data.Dataset):
    def __init__(self, df, reweight='none', lds=False, lds_kernel='gaussian', lds_ks=9, lds_sigma=1, labeled=True, num_slices=None):
        self.df = df
        self.weights = prepare_weights(df, reweight, lds, lds_kernel, lds_ks, lds_sigma)
        self.labeled = labeled
        self.num_slices = num_slices
        self.samples_per_img = (130 - num_slices + 1) if num_slices else 1

    def __getitem__(self, idx):
        img_idx = idx // self.samples_per_img
        chunk_idx = idx % self.samples_per_img

        row = self.df.iloc[img_idx]
        image = nibabel.load(row['path']).get_fdata()
        image = image[54:184, 25:195, 12:132] # Crop out zeroes
        image /= np.percentile(image, 95) # Normalize intensity

        if self.num_slices:
            image = image[chunk_idx:(chunk_idx+self.num_slices), :, :]

        if self.labeled:
            age = row['age']
            weight = self.weights[img_idx] if self.weights is not None else 1.
            return (image, age, weight)
        else:
            return image

    def __len__(self):
        return len(self.df) * self.samples_per_img
