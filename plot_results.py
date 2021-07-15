import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=str)
    opts = parser.parse_args()

    results_dir = os.path.join(SCRIPT_DIR, "results", opts.run)
    figures_dir = os.path.join(SCRIPT_DIR, "figures", opts.run)
    os.makedirs(figures_dir, exist_ok=True)

    train_losses = np.loadtxt(f"{results_dir}/train_losses_over_time.txt")
    val_losses = np.loadtxt(f"{results_dir}/val_losses_over_time.txt")
    with open(f"{results_dir}/best_model_val_losses.txt") as f:
        val_losses_per_bin = json.load(f)
    with open(f"{results_dir}/config.json") as f:
        cfg = json.load(f)

    cfg_desc = f"{cfg['arch']} / {cfg['sample']} / {cfg['reweight']}"
    if cfg['lds']:
        cfg_desc += f" (lds: {cfg['lds_kernel']} / {cfg['lds_ks']} / {cfg['lds_sigma']})"

    n_epochs = len(train_losses)
    plt.title(cfg_desc)
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.plot(range(n_epochs), train_losses, label="Training loss")
    plt.plot(range(n_epochs), val_losses, label="Validation loss")
    plt.legend()
    plt.savefig(f"{figures_dir}/train_and_val_losses.png")
    plt.clf()

    bins = val_losses_per_bin.keys()
    plt.title(cfg_desc)
    plt.xlabel("Age bin")
    plt.ylabel("MAE")
    plt.xticks(rotation='vertical')
    plt.bar(bins, val_losses_per_bin)
    plt.savefig(f"{figures_dir}/best_model_val_losses.png")
    plt.clf()

if __name__ == '__main__':
    main()
