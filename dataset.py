import nibabel
import numpy as np
from scipy.ndimage import convolve1d
from torch.utils import data

from lds import get_lds_kernel_window

def prepare_weights(df, reweight, lds, lds_kernel, lds_ks, lds_sigma):
    if reweight == 'none':
        return None
    
    # todo: clipping
    bin_counts = df['agebin'].value_counts()
    if reweight == 'inv':
        num_per_label = [bin_counts[bin] for bin in df['agebin']]
    elif reweight == 'sqrt_inv':
        num_per_label = [np.sqrt(bin_counts[bin]) for bin in df['agebin']]
    
    if lds:
        lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
        smoothed_value = convolve1d(
            np.asarray([v for _, v in bin_counts.items()]), weights=lds_kernel_window, mode='constant')
        num_per_label = [smoothed_value[bin] for bin in df['agebin']]

    weights = [1. / x for x in num_per_label]
    scaling = len(weights) / np.sum(weights)
    weights = [scaling * x for x in weights]
    return weights

class AgePredictionDataset(data.Dataset):
    def __init__(self, df, reweight='none', lds=False, lds_kernel='gaussian', lds_ks=9, lds_sigma=1):
        self.df = df
        self.weights = prepare_weights(df, reweight, lds, lds_kernel, lds_ks, lds_sigma)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = nibabel.load(row['path']).get_fdata()
        image = image[54:184, 25:195, 12:132] # Crop out zeroes
        image /= np.percentile(image, 95) # Normalize intensity
        age = row['age']
        weight = self.weights[idx] if self.weights is not None else 1.
        return (image, age, weight)

    def __len__(self):
        return len(self.df)
