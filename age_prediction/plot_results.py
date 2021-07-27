import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=str)
    opts = parser.parse_args()

    results_dir = os.path.join(ROOT_DIR, "results", opts.run)
    figures_dir = os.path.join(ROOT_DIR, "figures", opts.run)
    os.makedirs(figures_dir, exist_ok=True)

    train_losses = np.loadtxt(f"{results_dir}/train_losses_over_time.txt")
    val_losses = np.loadtxt(f"{results_dir}/val_losses_over_time.txt")
    best_model_val_preds = pd.read_csv(f"{results_dir}/best_model_val_preds.csv")
    with open(f"{results_dir}/config.json") as f:
        cfg = json.load(f)

    cfg_desc = f"arch: {cfg['arch']} / sample: {cfg['sample']} / reweight: {cfg['reweight']}"
    if cfg['lds']:
        cfg_desc += " + lds"

    n_epochs = len(train_losses)
    plt.title(cfg_desc)
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.xticks(rotation='vertical')
    plt.plot(range(n_epochs), train_losses, label="Training loss")
    plt.plot(range(n_epochs), val_losses, label="Validation loss")
    plt.legend()
    plt.savefig(f"{figures_dir}/train_and_val_losses.png")
    plt.clf()

    def mae(bin):
        subjects_in_bin = best_model_val_preds[best_model_val_preds['agebin'] == bin]
        return np.mean(np.abs(subjects_in_bin['age'] - subjects_in_bin['age_pred']))

    bins = sorted(set(best_model_val_preds['agebin']))
    best_model_val_losses_per_bin = [mae(bin) for bin in bins]
    plt.title(cfg_desc)
    plt.xlabel("Age bin")
    plt.ylabel("MAE")
    plt.xticks(rotation='vertical')
    plt.bar(bins, best_model_val_losses_per_bin)
    plt.savefig(f"{figures_dir}/best_model_val_losses.png")
    plt.clf()

if __name__ == '__main__':
    main()
