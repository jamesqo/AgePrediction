import nibabel
import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d, zoom
from torch.utils import data

from .utils import get_lds_kernel_window

def prepare_weights(df, reweight, lds):
    if reweight == 'none':
        return None
    
    bin_counts = df['agebin'].value_counts()
    # num_per_label[i] = the number of subjects in the age bin of the ith subject in the dataset
    if reweight == 'inv':
        num_per_label = [bin_counts[bin] for bin in df['agebin']]
    elif reweight == 'sqrt_inv':
        num_per_label = [np.sqrt(bin_counts[bin]) for bin in df['agebin']]
    
    if lds:
        lds_kernel_window = get_lds_kernel_window(lds['kernel'], lds['ks'], lds['sigma'])
        smoothed_value = pd.Series(
            convolve1d(bin_counts.values, weights=lds_kernel_window, mode='constant'),
            index=bin_counts.index)
        num_per_label = [smoothed_value[bin] for bin in df['agebin']]

    weights = [1. / x for x in num_per_label]
    scaling = len(weights) / np.sum(weights)
    weights = [scaling * x for x in weights]
    return weights

class AgePredictionDataset(data.Dataset):
    def __init__(self, df, reweight='none', lds=None, labeled=True, ravens=False, zoom=False):
        self.df = df
        self.weights = prepare_weights(df, reweight, lds)
        self.labeled = labeled
        self.ravens = ravens
        self.zoom = zoom

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = nibabel.load(row['path']).get_fdata()
        image = image[54:184, 25:195, 12:132] # Crop out zeroes
        image /= np.percentile(image, 95) # Normalize intensity
        if self.zoom:
            factor = (96/130, 96/170, 96/120)
            image = zoom(image, zoom=factor)
            assert image.shape == (96,96,96)

        if self.labeled:
            age = row['age']
            weight = self.weights[idx] if self.weights is not None else 1.

        if self.ravens:
            ravens_image = nibabel.load(row['ravens_path']).get_fdata()
            ravens_image /= 10_000 # Normalization
            if self.zoom:
                factor = (96/ravens_image.shape[0], 96/ravens_image.shape[1], 96/ravens_image.shape[2])
                ravens_image = zoom(ravens_image, zoom=factor)
                assert ravens_image.shape == (96,96,96)
        
        if self.ravens:
            return (image, ravens_image, age, weight) if self.labeled else (image, ravens_image)
        else:
            return (image, age, weight) if self.labeled else image

    def __len__(self):
        return len(self.df)
