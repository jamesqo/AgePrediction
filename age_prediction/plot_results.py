import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")

def has_results(job_id):
    required_files = [
        #f"{RESULTS_DIR}/{job_id}/train_losses_over_time.txt",
        #f"{RESULTS_DIR}/{job_id}/val_losses_over_time.txt",
        f"{RESULTS_DIR}/{job_id}/best_model_val_preds.csv",
        f"{RESULTS_DIR}/{job_id}/config.json"
    ]
    result = all([os.path.isfile(f) for f in required_files])
    if not result:
        print(f"Job {job_id} is missing 1 or more required file(s), skipping")
    return result

def describe_strat(cfg):
    if cfg['sample'] != 'none':
        return cfg['sample']
    elif cfg['reweight'] != 'none':
        desc = cfg['reweight']
        if cfg['lds']:
            desc += " + lds"
        if 'fds' in cfg and cfg['fds']:
            desc += " + fds"
        return desc
    return 'baseline'

def describe_job(cfg):
    return describe_strat(cfg) + f" ({cfg['arch']})"

def load_results():
    all_results = {}

    with open(f"{ROOT_DIR}/bookkeeping.yaml", 'r') as f:
        jobs = yaml.safe_load(f)

        for (arch, d) in jobs.items():
            for (strat, job_id) in d.items():
                if not has_results(job_id):
                    continue

                val_preds = pd.read_csv(f"{RESULTS_DIR}/{job_id}/best_model_val_preds.csv")
                with open(f"{RESULTS_DIR}/{job_id}/config.json") as f:
                    cfg = json.load(f)
                
                #job_desc = describe_job(cfg)
                job_desc = f"{strat} ({arch})"
                results = {
                    'val_preds': val_preds,
                    'cfg': cfg
                }
                all_results[job_desc] = results
    
    return all_results

def plot_losses_over_time(results, fname):
    train_losses = results['train_losses']
    val_losses = results['val_losses']
    cfg = results['config']
    n_epochs = len(train_losses)
    
    title = describe_job(cfg)

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

def plot_val_losses(all_results, arch, job_descs, fname, sampling=None):
    MAX_BIN = 17 # Maximum 5y age bin we're interested in (we ignore patients aged 90+)
    BIN_WIDTH = 5 # Width of each bin for plotting purposes
    MAX_AGE = (MAX_BIN+1)*BIN_WIDTH - 1
    TRAIN_VAL_RATIO = 4. # Ratio of training set size to validation set size (it was an 80%/20% split)

    LOSS_MAX_BIN = 9

    if '-' in arch:
        subname = arch[arch.index('-')+1:]
        base_arch = arch[:arch.index('-')]
        pred_key = f'age_pred_{subname}'
    else:
        subname = None
        base_arch = arch
        pred_key = 'age_pred'

    def mae(bin, df, bin_width=BIN_WIDTH):
        subjects_in_bin = df[(df['age'] // bin_width) == bin]
        assert len(subjects_in_bin) > 0
        return np.mean(np.abs(subjects_in_bin['age'] - subjects_in_bin[pred_key]))
    
    # Setup the plot axes
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    display_arch = {
        'resnet18': "ResNet-18",
        'vgg8': "VGG8",
        'sfcn': "SFCN",
        'glt': "GLT",
        'relnet-sum': "RelationNet-sum",
        'relnet-max': "RelationNet-max",
        'relnet-min': "RelationNet-min",
        '3dt': "3D Transformer"
    }[arch]
    ax.set_title(f"Validation losses per 5-year age bin ({display_arch})")
    ax.set_xlabel("Age bin")
    ax.set_ylabel("MAE")
    ax.set_ylim(0, 15)
    #ax2.set_ylabel("Number of samples", rotation=270)
    ax2.set_ylabel("Number of training samples")
    ax2.set_ylim(0, 1000)
    
    # Calculate the bin counts of the train df (this should be the same across all models)
    bins = np.arange(MAX_BIN + 1)
    display_bins = [bin*BIN_WIDTH for bin in bins]
    display_bins_incl = display_bins + [MAX_AGE+1]
    centroids = [db + (BIN_WIDTH/2) for db in display_bins]
    val_df = list(all_results.values())[0]['val_preds']
    train_bin_counts = [TRAIN_VAL_RATIO * sum((val_df['age'] // BIN_WIDTH) == bin) for bin in bins]

    job_descs.insert(0, ('baseline', "Baseline", '#888888', '-')) # Show the curve for the baseline model in every plot
    for i, (desc, label, color, style) in enumerate(job_descs):
        cfg = f"{desc} ({base_arch})"
        display_cfg = f"{desc} ({arch})"
        if cfg not in all_results:
            print(f"Couldn't find {display_cfg} in results, skipping")
            print("======")
            continue

        val_df = all_results[cfg]['val_preds']

        # Calculate and print the val MAE across all predictions
        overall_mae = np.mean(np.abs(val_df['age'] - val_df[pred_key]))
        print(f"Overall MAE for {display_cfg}: {overall_mae:.3f}")

        # Smooth val losses using a Gaussian kernel, and plot them
        val_losses_per_bin = np.array([mae(bin, val_df) for bin in bins])
        smoothed_losses = np.zeros(val_losses_per_bin.shape)
        sigma = 1.
        for bin in bins:
            kernel = np.exp(-(bins - bin) ** 2 / (2 * sigma ** 2))
            kernel = kernel / sum(kernel)
            smoothed_losses[bin] = sum(val_losses_per_bin * kernel)

        ax.set_xticks(display_bins_incl)
        ax.plot(centroids, smoothed_losses, label=label, color=color, linestyle=style)
        ax.set_zorder(10)
        ax.patch.set_alpha(0.)
        
        # Calculate Pearson correlation between val losses / train bin counts (per 5y bin) and annotate the plot with it
        pcorr, _ = stats.pearsonr(val_losses_per_bin, train_bin_counts)
        print(f"ρ for {display_cfg}: {pcorr:.3f}")
        anno_x = [15, 30, 45, 60, 75][i] # X location of annotation
        anno_y = smoothed_losses[anno_x//BIN_WIDTH]-.5 # Y location of annotation
        ax.annotate(f"ρ = {pcorr:.3f}", (anno_x, anno_y), color=color)

        # Calculate the validation loss per 1-year age bin.
        # Group the losses themselves into 1y bins, and report their entropy.
        val_bins_1y = [bin for bin in np.arange(MAX_AGE+1) if bin in set(val_df['age'] // 1)]
        val_losses_per_1y_bin = np.array([mae(bin, val_df, bin_width=1) for bin in val_bins_1y])
        val_loss_counts = {}
        for loss_bin in range(LOSS_MAX_BIN+1):
            val_loss_counts[loss_bin] = sum(
                loss >= loss_bin and loss < (loss_bin+1) for loss in val_losses_per_1y_bin)
        val_loss_counts[LOSS_MAX_BIN+1] = sum(loss >= (LOSS_MAX_BIN+1) for loss in val_losses_per_1y_bin)
        n_losses = len(val_bins_1y)
        ps = [count / n_losses for count in val_loss_counts.values()]
        entropy = sum([0 if p == 0 else (-p * np.log(p)) for p in ps])
        print(f"Entropy for {display_cfg}: {entropy:.3f}")

        # Report the final criterion (MAE*entropy)
        criterion = overall_mae*entropy
        print(f"Criterion (=MAE*entropy) for {display_cfg}: {criterion:.3f}")

        print("======")
    
    # Add legend
    l = ax.legend()
    l.set_zorder(20)
    x = display_bins
    weights = train_bin_counts
    color = '#0f0f0f30'
    if sampling == 'under':
        USAMP_LIMIT = 18
        x = [x, x]
        under_counts = [USAMP_LIMIT*5 for bin in bins]
        under_counts[-1] = USAMP_LIMIT*3
        weights = [weights, under_counts]
        color = [color, 'pink']
    elif sampling == 'over':
        OSAMP_LIMIT = 126
        x = [x, x]
        over_counts = [OSAMP_LIMIT*5 for bin in bins]
        over_counts[-1] = OSAMP_LIMIT*3
        weights = [weights, over_counts]
        color = [color, '#87cefa80']
    ax2.hist(x, bins=display_bins_incl, weights=weights, color=color, stacked=False)

    fig.savefig(f"{FIGURES_DIR}/{fname}")
    plt.close(fig)

def main():
    shutil.rmtree(FIGURES_DIR)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    all_results = load_results()

    plt.rcParams.update({'font.family': 'C059'})

    '''
    ## Plot losses over time
    for results in all_results.values():
        job_id = results['config']['job_id']
        plot_losses_over_time(results, f"losses_over_time_{job_id}.png")
    '''
    
    ## Plot validation losses per bin
    for arch in ('resnet18', 'vgg8', 'sfcn', 'glt', '3dt'):
        plot_val_losses(all_results, arch,
            [
                ('under', "Undersampling", 'red', '-'),
                ('scale-down', "Scaling down", 'red', '--'),
            ],
            f"val_losses_{arch}_undersampling.png", sampling='under')
        plot_val_losses(all_results, arch,
            [
                ('over', "Oversampling", 'royalblue', '-'),
                ('scale-up', "Scaling up", 'royalblue', '--')
            ],
            f"val_losses_{arch}_oversampling.png", sampling='over')
        plot_val_losses(all_results, arch,
            [
                ('inv', "Inverse weighting", 'red', '-'),
                ('lds+inv', "Inverse weighting + LDS", 'royalblue', '-'),
                ('fds+inv', "Inverse weighting + FDS", 'orange', '-'),
                ('lds+fds+inv', "Inverse weighting + LDS + FDS", 'green', '-')
            ],
            f"val_losses_{arch}_reweighting_inv.png")
        plot_val_losses(all_results, arch,
            [
                ('sqrt_inv', "Square-root inverse weighting", 'red', '-'),
                ('lds+sqrt_inv', "Square-root inverse weighting + LDS", 'royalblue', '-'),
                ('fds+sqrt_inv', "Square-root inverse weighting + FDS", 'orange', '-'),
                ('lds+fds+sqrt_inv', "Square-root inverse weighting + LDS + FDS", 'green', '-')
            ],
            f"val_losses_{arch}_reweighting_sqrt_inv.png")
    
    ## Plot histograms for each dataset + the combined dataset
    path = os.path.join(RESULTS_DIR, "sample-none", "merged_df.csv")
    df = pd.read_csv(path)
    
    label_locs = np.arange(len(set(df['dataset']))) - 0.5
    plt.hist(df['dataset'], bins=label_locs)
    plt.title("Number of samples contributed by each dataset")
    plt.xlabel("Dataset")
    plt.ylabel("Number of samples")
    plt.savefig(os.path.join(FIGURES_DIR, 'dataset_counts.png'))
    plt.clf()

    dataset_names = sorted(set(df['dataset']))
    dataset_names += ['combined']
    for dataset in dataset_names:
        subdf = df if dataset == 'combined' else df[df['dataset'] == dataset]
        bins = np.arange(18)
        display_bins = [5*bin for bin in bins]
        display_dataset = {
            'ABIDE_I': 'ABIDE-I',
            'beijingEn': 'BeijingEN',
            'BGSP': 'BGSP',
            'DLBS': 'DLBS',
            'IXI_600': 'IXI',
            'MGHBCH': 'MGHBCH',
            'NIH_PD': 'NIH-PD',
            'OASIS_3': 'OASIS-3',
            'combined': 'Combined'
        }[dataset]
        bin_counts = [sum((subdf['age'] // 5) == bin) for bin in bins]

        plt.hist(display_bins, bins=display_bins, weights=bin_counts)
        plt.xticks(display_bins)
        plt.title(display_dataset)
        plt.xlabel("Age")
        plt.ylabel("Number of samples")
        plt.savefig(os.path.join(FIGURES_DIR, f"bin_counts_{dataset}.png"))
        plt.clf()

    print("Done")

if __name__ == '__main__':
    main()
