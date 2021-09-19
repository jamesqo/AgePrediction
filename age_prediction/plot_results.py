import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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

def plot_val_losses(all_results, arch, job_descs, fname):
    def mae(bin, df):
        subjects_in_bin = df[(df['age'] // 5) == bin]
        assert len(subjects_in_bin) > 0
        return np.mean(np.abs(subjects_in_bin['age'] - subjects_in_bin['age_pred']))
    
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    display_arch = {'resnet18': "ResNet-18", 'vgg8': "VGG8", 'sfcn': "SFCN"}[arch]
    ax.set_title(f"Validation losses per 5-year age bin ({display_arch})")
    ax.set_xlabel("Age bin")
    ax.set_ylabel("MAE")
    ax.set_ylim(0, 15)
    #ax2.set_ylabel("Number of samples", rotation=270)
    ax2.set_ylabel("Number of samples")
    ax2.set_ylim(0, 250)
    
    bins = np.arange(18)
    display_bins = [5*bin for bin in bins]
    df = list(all_results.values())[0]['val_preds']
    bin_counts = [sum((df['age'] // 5) == bin) for bin in bins]

    job_descs.insert(0, ('none / none', "Baseline"))
    for i, (desc, label) in enumerate(job_descs):
        key = f"{arch} / {desc}"
        df = all_results[key]['val_preds']

        overall_mae = np.mean(np.abs(df['age'] - df['age_pred']))
        print(f"Overall MAE for {key}: {overall_mae:.3f}")

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
        color = '#888888' if is_baseline else ['red', 'royalblue'][i-1]
        ax.plot(display_bins, smoothed_losses, label=label, color=color)
        
        pcorr, _ = stats.pearsonr(losses_per_bin, bin_counts)
        print(f"ρ for {key}: {pcorr:.3f}")
        anno_x = [15, 30, 45][i]
        anno_y = smoothed_losses[anno_x//5]-.5
        ax.annotate(f"ρ = {pcorr:.3f}", (anno_x, anno_y), color=color)

        print()
    
    ax.legend()
    bin_counts = [sum((df['age'] // 5) == bin) for bin in bins]
    ax2.hist(display_bins, bins=display_bins, weights=bin_counts, color='#0f0f0f30')
    fig.savefig(f"{FIGURES_DIR}/{fname}")
    plt.close(fig)

def main():
    shutil.rmtree(FIGURES_DIR)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    all_results = load_results()

    plt.rcParams.update({'font.family': 'C059'})

    for results in all_results.values():
        job_id = results['config']['job_id']
        plot_losses_over_time(results, f"losses_over_time_{job_id}.png")
    
    for arch in ('resnet18', 'vgg8', 'sfcn'):
        plot_val_losses(all_results, arch, [('under / none', "Undersampling"), ('scale-down / none', "Scaling down")], f"val_losses_{arch}_under.png")
        plot_val_losses(all_results, arch, [('over / none', "Oversampling"), ('scale-up / none', "Scaling up")], f"val_losses_{arch}_over.png")
        plot_val_losses(all_results, arch, [('none / inv', "Inverse weighting"), ('none / inv + lds', "Inverse weighting + LDS")], f"val_losses_{arch}_inv.png")
        plot_val_losses(all_results, arch, [('none / sqrt_inv', "Square root-inverse weighting"), ('none / sqrt_inv + lds', "Square root-inverse weighting + LDS")], f"val_losses_{arch}_sqrt_inv.png")
    
    for mode in ('none',):
        path = os.path.join(RESULTS_DIR, f"sample-{mode}", "merged_df.csv")
        df = pd.read_csv(path)
        
        #print(df['dataset'].value_counts())
        plt.hist(df['dataset'], bins=np.arange(len(set(df['dataset'])))-0.5)
        plt.title("Number of samples contributed by each dataset")
        plt.xlabel("Dataset name")
        plt.ylabel("Number of samples")
        plt.savefig(os.path.join(FIGURES_DIR, 'dataset_counts.png'))
        plt.clf()

        for dataset in sorted(set(df['dataset'])) + ['Combined']:
            if dataset == 'Combined':
                subdf = df
            else:
                subdf = df[df['dataset'] == dataset]
            bins = np.arange(18)
            display_bins = [5*bin for bin in bins]
            bin_counts = [sum((subdf['age'] // 5) == bin) for bin in bins]
            plt.hist(display_bins, bins=display_bins, weights=bin_counts)

            plt.xticks(display_bins)
            plt.title(dataset) # todo: change to a user-friendly name
            plt.xlabel("Age")
            plt.ylabel("Number of samples")
            plt.savefig(os.path.join(FIGURES_DIR, f"bin_counts_{dataset}.png"))
            plt.clf()

    print("Done")

if __name__ == '__main__':
    main()
