import json
import os
import shutil

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

def describe_job(**cfg):
    desc = f"{cfg['arch']} / {cfg['sample']} / {cfg['reweight']}"
    if cfg['lds']:
        desc += " + lds"
    return desc

def load_results():
    job_ids = os.listdir(RESULTS_DIR)
    job_ids = list(filter(has_results, job_ids))

    all_results = {}
    for job_id in job_ids:
        train_losses = np.loadtxt(f"{RESULTS_DIR}/{job_id}/train_losses_over_time.txt")
        val_losses = np.loadtxt(f"{RESULTS_DIR}/{job_id}/val_losses_over_time.txt")
        val_preds = pd.read_csv(f"{RESULTS_DIR}/{job_id}/best_model_val_preds.csv")
        with open(f"{RESULTS_DIR}/{job_id}/config.json") as f:
            cfg = json.load(f)
        
        job_desc = describe_job(**cfg)
        results = {}
        results['train_losses'] = train_losses
        results['val_losses'] = val_losses
        results['val_preds'] = val_preds
        results['config'] = cfg
        all_results[job_desc] = results
    
    return all_results

def plot_losses_over_time(results, fname):
    train_losses = results['train_losses']
    val_losses = results['val_losses']
    cfg = results['config']
    n_epochs = len(train_losses)
    
    title = describe_job(**cfg)

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE")
    ax.set_ylim(0, 25)
    ax.plot(range(n_epochs), train_losses, label="Training loss")
    ax.plot(range(n_epochs), val_losses, label="Validation loss")
    ax.legend()
    fig.savefig(f"{FIGURES_DIR}/{fname}")
    plt.close(fig)

def plot_val_losses(all_results, arch, job_descs, fname, include_baseline=True):
    def mae(bin, df):
        subjects_in_bin = df[(df['age'] // 5) == bin]
        assert len(subjects_in_bin) > 0
        return np.mean(np.abs(subjects_in_bin['age'] - subjects_in_bin['age_pred']))
    
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.set_title(f"Validation losses per 5-year age bin ({arch})")
    ax.set_xlabel("Age bin")
    ax.set_ylabel("MAE")
    ax.set_ylim(0, 15)
    #ax2.set_ylabel("Number of samples", rotation=270)
    ax2.set_ylabel("Number of samples")
    ax2.set_ylim(0, 250)
    
    bins = np.arange(18)
    display_bins = [5*bin for bin in bins]

    if include_baseline:
        job_descs += ['none / none']
    for desc in job_descs:
        key = f"{arch} / {desc}"
        df = all_results[key]['val_preds']
        losses_per_bin = np.array([mae(bin, df) for bin in bins])
        
        # Smooth losses using a Gaussian kernel
        smoothed_losses = np.zeros(losses_per_bin.shape)
        sigma = 1.
        for bin in bins:
            kernel = np.exp(-(bins - bin) ** 2 / (2 * sigma ** 2))
            kernel = kernel / sum(kernel)
            smoothed_losses[bin] = sum(losses_per_bin * kernel)

        ax.set_xticks(display_bins)
        is_baseline = (desc == 'none / none')
        ax.plot(display_bins, smoothed_losses, label=desc, color=('lightgray' if is_baseline else None))
    
    ax.legend()
    bin_counts = [sum((df['age'] // 5) == bin) for bin in bins]
    ax2.hist(display_bins, bins=display_bins, weights=bin_counts, color='#0f0f0f80')
    fig.savefig(f"{FIGURES_DIR}/{fname}")
    plt.close(fig)

def main():
    shutil.rmtree(FIGURES_DIR)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    all_results = load_results()

    for results in all_results.values():
        job_id = results['config']['job_id']
        plot_losses_over_time(results, f"losses_over_time_{job_id}.png")

    for arch in ('resnet18', 'vgg8'):
        plot_val_losses(all_results, arch, ['under / none', 'scale-down / none'], f"val_losses_{arch}_under.png")
        plot_val_losses(all_results, arch, ['over / none', 'scale-up / none'], f"val_losses_{arch}_over.png")
        plot_val_losses(all_results, arch, ['none / inv', 'none / inv + lds'], f"val_losses_{arch}_inv.png")
        plot_val_losses(all_results, arch, ['none / sqrt_inv', 'none / sqrt_inv + lds'], f"val_losses_{arch}_sqrt_inv.png")

    print("Done")

if __name__ == '__main__':
    main()
