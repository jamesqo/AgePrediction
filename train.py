import argparse
from datetime import datetime
import glob
import json
import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from torchvision import models

from dataset import AgePredictionDataset
from vgg import VGG8

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SPLITS_DIR = os.path.join(SCRIPT_DIR, "folderlist")
START_TIME = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = os.path.join(SCRIPT_DIR, "logs", f"{START_TIME}.log")

def log(message):
    with open(LOG_FILE, 'a+') as log_file:
        log_file.write(f"{message}\n")

def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('arch', type=str)

    parser.add_argument('--batch-size', type=int, default=5, help='batch size')
    parser.add_argument('--n-epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--initial-lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--step-size', type=int, default=10, help='learning rate decay period')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor')
    parser.add_argument('--weight-decay', type=float, default=1e-6, help='weight decay')

    parser.add_argument('--sample', type=str, choices=['none', 'over', 'under', 'scale-up', 'scale-down'], default='none', help='sampling strategy')
    parser.add_argument('--reweight', type=str, choices=['none', 'inv', 'sqrt_inv'], default='none', help='reweighting strategy')

    parser.add_argument('--lds', action='store_true', default=False, help='whether to enable LDS')
    parser.add_argument('--lds_kernel', type=str, default='gaussian',
                        choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
    parser.add_argument('--lds_ks', type=int, default=9, help='LDS kernel size: should be odd number')
    parser.add_argument('--lds_sigma', type=float, default=1, help='LDS gaussian/laplace kernel sigma')

    ## For testing purposes only
    parser.add_argument('--cpu', action='store_true', help='use CPU instead of GPU')
    parser.add_argument('--eval', type=str, help='evaluate a pretrained model')
    parser.add_argument('--max-samples', type=int, help='limit the number of samples used for training/validation')

    args = parser.parse_args()
    assert (args.sample == 'none') or (args.reweight == 'none'), "--sample is incompatible with --reweight"

    for arg in vars(args):
        log(f"{arg}: {getattr(args, arg)}")
    log('='*20)
    return args

def load_samples(split_fnames, max_samples=None):
    def correct_path(path):
        path = path.replace('/ABIDE/', '/ABIDE_I/')
        path = path.replace('/NIH-PD/', '/NIH_PD/')
        return path

    schema = {'id': str, 'age': float, 'sex': str, 'path': str}
    dfs = []
    for fname in split_fnames:
        split_num = int(fname[:-len('.list')].split('_')[-1])

        df = pd.read_csv(fname, sep=' ', header=None, names=['id', 'age', 'sex', 'path'], dtype=schema)
        df['path'] = df['path'].apply(correct_path)
        df['agebin'] = df['age'] // 5
        df['split'] = split_num
        dfs.append(df)
    
    combined_df = pd.concat(dfs, axis=0)
    
    if max_samples is not None:
        combined_df = combined_df.sample(max_samples)
    
    return combined_df

def resample(df, mode):
    def count_samples(bin):
        return sum(df['agebin'] == bin)
    
    log("Resampling training data")

    bins = sorted(set(df['agebin']))
    bin_counts = [count_samples(bin) for bin in bins]
    bin_ratios = [count / df.shape[0] for count in bin_counts]
    log(f"Bin counts: {dict(zip(bins, bin_counts))}")

    max_bin = max(bins, key=count_samples)
    max_count = count_samples(max_bin)
    log(f"Max bin is {max_bin} with {max_count} samples")

    # The actual min agebin could have a very small number of samples, so instead
    # use the smallest one with a sample count >= the 10th percentile.
    min_min_count = np.percentile(bin_counts, 10)
    log(f"10th percentile of bin counts: {min_min_count}")
    min_bin = min([bin for bin in bins if count_samples(bin) >= min_min_count], key=count_samples)
    min_count = count_samples(min_bin)
    log(f"Min bin is {min_bin} with {min_count} samples")

    if mode == 'none':
        pass
    elif mode == 'over':
        for bin in bins:
            n_under = max_count - count_samples(bin)
            if n_under == 0:
                continue
            new_samples = df[df['agebin'] == bin].sample(n_under, replace=True)
            df = pd.concat([df, new_samples], axis=0)
    elif mode == 'under':
        for bin in bins:
            n_over = count_samples(bin) - min_count
            if n_over <= 0:
                continue
            new_samples = df[df['agebin'] == bin].sample(min_count, replace=False)
            df = df[df['agebin'] != bin]
            df = pd.concat([df, new_samples], axis=0)
    elif mode == 'scale-up':
        target_count = max_count * len(bins)
        for bin in bins:
            count = int(bin_ratios[int(bin)] * target_count)
            new_samples = df[df['agebin'] == bin].sample(count, replace=True)
            df = df[df['agebin'] != bin]
            df = pd.concat([df, new_samples], axis=0)
    elif mode == 'scale-down':
        target_count = min_count * len(bins)
        for bin in bins:
            count = int(bin_ratios[int(bin)] * target_count)
            new_samples = df[df['agebin'] == bin].sample(count, replace=False)
            df = df[df['agebin'] != bin]
            df = pd.concat([df, new_samples], axis=0)
    else:
        raise Exception(f"Invalid sampling mode: {mode}")

    log(f"Number of samples in final training dataset: {df.shape[0]}")
    return df

def train(model, optimizer, train_loader, device):
    def weighted_l1_loss(inputs, targets, weights):
        loss = F.l1_loss(inputs, targets, reduction='none')
        loss *= weights.expand_as(loss)
        loss = torch.mean(loss)
        return loss

    model.train()

    losses = []

    for batch_idx, (images, ages, weights) in enumerate(train_loader):
        images, ages, weights = images.to(device), ages.to(device), weights.to(device)
        optimizer.zero_grad()
        age_preds = model(images).view(-1)
        loss = weighted_l1_loss(age_preds, ages, weights)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if batch_idx % 10 == 0:
            log(f"Batch {batch_idx} loss {loss} mean loss {np.mean(losses)}")
    
    return np.mean(losses)

def validate(model, val_loader, device):
    model.eval()

    losses = []

    with torch.no_grad():
        for (images, ages, _) in val_loader:
            images, ages = images.to(device), ages.to(device)
            age_preds = model(images).view(-1)
            loss = F.l1_loss(age_preds, ages, reduction='mean')

            losses.append(loss.item())
    
    return np.mean(losses)

def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    opts = parse_options()

    checkpoint_dir = os.path.join(SCRIPT_DIR, "checkpoints", opts.eval or START_TIME)
    results_dir = os.path.join(SCRIPT_DIR, "results", START_TIME)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    log("Setting up dataset")

    split_fnames = glob.glob(f"{SPLITS_DIR}/nfold_imglist_all_nfold_*.list")
    assert len(split_fnames) == 5

    df = load_samples(split_fnames, opts.max_samples)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['agebin'])
    train_df = resample(train_df, mode=opts.sample)
    train_dataset = AgePredictionDataset(train_df, reweight=opts.reweight, lds=opts.lds, lds_kernel=opts.lds_kernel, lds_ks=opts.lds_ks, lds_sigma=opts.lds_sigma)
    val_dataset = AgePredictionDataset(val_df)

    train_loader = data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=opts.batch_size)

    log("Setting up model")

    if opts.arch == 'resnet18':
        model = models.resnet18(num_classes=1)
        # Set the number of input channels to 130
        model.conv1 = nn.Conv2d(130, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif opts.arch == 'vgg8':
        model = VGG8(in_channels=130, num_classes=1)
    else:
        raise Exception(f"Invalid arch: {opts.arch}")
    model.double()
    device = torch.device('cpu' if opts.cpu else 'cuda')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=opts.initial_lr, weight_decay=opts.weight_decay)
    scheduler = StepLR(optimizer, step_size=opts.step_size, gamma=opts.gamma)

    if opts.eval is None:

        ## Training process

        log("Starting training process")

        best_epoch = None
        best_val_loss = np.inf
        train_losses = []
        val_losses = []

        for epoch in range(opts.n_epochs):
            log(f"Epoch {epoch}/{opts.n_epochs}")
            train_loss = train(model, optimizer, train_loader, device)
            log(f"Mean training loss: {train_loss}")
            val_loss = validate(model, val_loader, device)
            log(f"Mean validation loss: {val_loss}")
            scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if val_loss < best_val_loss:
                torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pth")

                best_epoch = epoch
                best_val_loss = val_loss
        
        log(f"Best model had validation loss of {best_val_loss}, occurred at epoch {best_epoch}")
    
    ## Evaluation

    log("Evaluating best model on val dataset")
   
    checkpoint = torch.load(f"{checkpoint_dir}/best_model.pth")
    model.load_state_dict(checkpoint)
    
    bins = sorted(set(df['agebin']))
    bin_losses = []
    for bin in bins:
        bin_df = val_df[val_df['agebin'] == bin]
        bin_dataset = AgePredictionDataset(bin_df)
        bin_loader = data.DataLoader(bin_dataset, batch_size=opts.batch_size)
        
        bin_loss = validate(model, bin_loader, device)
        bin_losses.append(bin_loss)
    
    ## Save results so we can plot them later

    log("Saving results")
    
    with open(f"{results_dir}/config.json", 'w+') as cfg_file:
        cfg = vars(opts)
        cfg['start_time'] = START_TIME
        cfg['train_counts'] = json.loads(train_df['agebin'].value_counts().to_json())
        cfg['val_counts'] = json.loads(val_df['agebin'].value_counts().to_json())
        json.dump(cfg, cfg_file, sort_keys=True, indent=4)
    
    if opts.eval is None:
        np.savetxt(f"{results_dir}/train_losses_during_training.txt", train_losses)
        np.savetxt(f"{results_dir}/val_losses_during_training.txt", val_losses)
    np.savetxt(f"{results_dir}/best_model_val_losses_per_bin.txt", bin_losses)
    

if __name__ == '__main__':
    main()
