import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")

def has_results(job_id):
    return os.path.isfile(f"{RESULTS_DIR}/{job_id}/train_losses_over_time.txt") and \
           os.path.isfile(f"{RESULTS_DIR}/{job_id}/val_losses_over_time.txt") and \
           os.path.isfile(f"{RESULTS_DIR}/{job_id}/best_model_val_preds.csv") and \
           os.path.isfile(f"{RESULTS_DIR}/{job_id}/config.json")

def plot_losses_over_time(train_losses, val_losses, description, job_id):
    n_epochs = len(train_losses)

    fig, ax = plt.subplots()
    ax.set_title(description)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax.plot(range(n_epochs), train_losses, label="Training loss")
    ax.plot(range(n_epochs), val_losses, label="Validation loss")
    ax.legend()
    fig.savefig(f"{FIGURES_DIR}/{job_id}_train_and_val_losses_over_time.png")

def plot_val_losses_per_5y_bin(ax, preds, description):
    def mae(bin):
        subjects_in_bin = preds[(preds['age'] // 5) == bin]
        assert len(subjects_in_bin) > 0
        return np.mean(np.abs(subjects_in_bin['age'] - subjects_in_bin['age_pred']))
    
    if description not in ('none / inv', 'none / inv + lds', 'under / none', 'none / none'):
        return

    bins = np.arange(18)
    losses_per_bin = np.array([mae(bin) for bin in bins])
    
    # Smooth losses using a Gaussian kernel
    smoothed_losses = np.zeros(losses_per_bin.shape)
    sigma = 1.
    for bin in bins:
        kernel = np.exp(-(bins - bin) ** 2 / (2 * sigma ** 2))
        kernel = kernel / sum(kernel)
        smoothed_losses[bin] = sum(losses_per_bin * kernel)

    display_bins = [5*bin for bin in bins]
    ax.set_xticks(display_bins)
    ax.plot(display_bins, smoothed_losses, label=description)

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    job_ids = os.listdir(RESULTS_DIR)
    job_ids = list(filter(has_results, job_ids))

    ## Plot training/validation losses over time

    for job_id in job_ids:
        train_losses = np.loadtxt(f"{RESULTS_DIR}/{job_id}/train_losses_over_time.txt")
        val_losses = np.loadtxt(f"{RESULTS_DIR}/{job_id}/val_losses_over_time.txt")
        with open(f"{RESULTS_DIR}/{job_id}/config.json") as f:
            cfg = json.load(f)

        description = f"{cfg['arch']} / {cfg['sample']} / {cfg['reweight']}"
        if cfg['lds']:
            description += " + lds"
        
        plot_losses_over_time(train_losses, val_losses, description, job_id)
    
    ## Plot validation losses per age bin

    archs = ['resnet18', 'vgg8']
    val_losses_figs = {}
    val_losses_axs = {}
    for arch in archs:
        fig, ax = plt.subplots()
        ax.set_title(f"Validation losses per 5-year age bin ({arch})")
        ax.set_xlabel("Age bin")
        ax.set_ylabel("MAE")
        ax.set_ylim(0, 15)
        val_losses_figs[arch] = fig
        val_losses_axs[arch] = ax

    for job_id in job_ids:
        val_preds = pd.read_csv(f"{RESULTS_DIR}/{job_id}/best_model_val_preds.csv")
        with open(f"{RESULTS_DIR}/{job_id}/config.json") as f:
            cfg = json.load(f)

        description = f"{cfg['sample']} / {cfg['reweight']}"
        if cfg['lds']:
            description += " + lds"
        
        arch = cfg['arch']
        plot_val_losses_per_5y_bin(val_losses_axs[arch], val_preds, description)

    for arch in archs:
        val_losses_axs[arch].legend()
        val_losses_figs[arch].savefig(f"{FIGURES_DIR}/{arch}_val_losses_per_bin.png")

    print("Done")

if __name__ == '__main__':
    main()
