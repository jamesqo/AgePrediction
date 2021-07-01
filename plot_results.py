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

    train_losses = np.loadtxt(f"{results_dir}/train_losses_during_training.txt")
    val_losses = np.loadtxt(f"{results_dir}/val_losses_during_training.txt")
    val_losses_per_bin = np.loadtxt(f"{results_dir}/best_model_val_losses_per_bin.txt")
    with open(f"{results_dir}/config.json") as cfg_file:
        cfg = json.load(cfg_file)

    cfg_desc = f"{cfg['arch']} / {cfg['sampling_mode']}"

    n_epochs = len(train_losses)
    plt.title(cfg_desc)
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.plot(range(n_epochs), train_losses, label="Training loss")
    plt.plot(range(n_epochs), val_losses, label="Validation loss")
    plt.legend()
    plt.savefig(f"{figures_dir}/train_and_val_losses_over_time.png")
    plt.clf()

    n_bins = len(val_losses_per_bin)
    bins = [f"{5*i}-{5*i+4}" for i in range(n_bins)]
    plt.title(cfg_desc)
    plt.xlabel("Age bin")
    plt.ylabel("MAE")
    plt.xticks(rotation='vertical')
    plt.bar(bins, val_losses_per_bin)
    plt.savefig(f"{figures_dir}/val_losses_per_bin.png")
    plt.clf()

if __name__ == '__main__':
    main()
