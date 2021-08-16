import argparse
from datetime import datetime
import glob
import json
import os
import random
import sys

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

from .dataset import AgePredictionDataset
from .sfcn import SFCN
from .vgg import VGG8
from .window import SlidingWindow

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SPLITS_DIR = os.path.join(ROOT_DIR, "folderlist")
START_TIME = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
LOG_FILE = ''

def log(message):
    if LOG_FILE == '':
        print(message)
        return

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    with open(LOG_FILE, 'a+') as log_file:
        log_file.write(f"{message}\n")

def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('arch', type=str, choices=['resnet18', 'vgg8', 'sfcn', 'resnet18-20s', 'vgg8-20s'])
    parser.add_argument('--job-id', type=str, required=True, help='SLURM job ID')

    parser.add_argument('--batch-size', type=int, default=5, help='batch size')
    parser.add_argument('--n-epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--initial-lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--step-size', type=int, default=10, help='learning rate decay period')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor')
    parser.add_argument('--weight-decay', type=float, default=1e-6, help='weight decay')

    parser.add_argument('--sample', type=str, choices=['none', 'over', 'under', 'scale-up', 'scale-down'], default='none', help='sampling strategy')
    parser.add_argument('--reweight', type=str, choices=['none', 'inv', 'sqrt_inv'], default='none', help='reweighting strategy')
    parser.add_argument('--bin-width', type=int, default=1, help='width of age bins')
    parser.add_argument('--min-bin-count', type=int, default=10, help='minimum number of samples per age bin (bins with fewer samples will be removed)')
    parser.add_argument('--oversamp-limit', type=int, default=126, help='maximum number of samples each age bin will be inflated to when oversampling (bins with more samples will be cut off)')

    parser.add_argument('--lds', action='store_true', default=False, help='whether to enable LDS')
    parser.add_argument('--lds_kernel', type=str, default='gaussian',
                        choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
    parser.add_argument('--lds_ks', type=int, default=9, help='LDS kernel size: should be odd number')
    parser.add_argument('--lds_sigma', type=float, default=1, help='LDS gaussian/laplace kernel sigma')

    ## For testing purposes only
    parser.add_argument('--cpu', action='store_true', help='use CPU instead of GPU')
    parser.add_argument('--eval', type=str, help='evaluate a pretrained model')
    parser.add_argument('--max-samples', type=int, help='limit the number of samples used for training/validation')
    parser.add_argument('--print-bin-counts', action='store_true', help='print age bin counts and exit')

    args = parser.parse_args()
    assert (args.sample == 'none') or (args.reweight == 'none'), "--sample is incompatible with --reweight"

    global LOG_FILE
    LOG_FILE = os.path.join(ROOT_DIR, "logs", f"{args.job_id}.log")
    try:
        os.remove(LOG_FILE)
    except FileNotFoundError:
        pass

    for arg in vars(args):
        log(f"{arg}: {getattr(args, arg)}")
    log('='*20)
    return args

def load_samples(split_fnames, bin_width=1, min_bin_count=10, max_samples=None):
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
        df['agebin'] = df['age'] // bin_width
        df['split'] = split_num
        dfs.append(df)
    
    combined_df = pd.concat(dfs, axis=0)

    if min_bin_count is not None:
        bin_counts = combined_df['agebin'].value_counts()
        bins_below_cutoff = [bin for bin in bin_counts.keys() if bin_counts[bin] < min_bin_count]
        combined_df = combined_df[~combined_df['agebin'].isin(bins_below_cutoff)]
    
    if max_samples is not None:
        combined_df = combined_df.sample(max_samples)
    
    return combined_df

def resample(df, mode, oversamp_limit):
    log("Resampling training data")

    bins = sorted(set(df['agebin']))
    bin_counts = {bin: sum(df['agebin'] == bin) for bin in bins}
    bin_ratios = {bin: count / df.shape[0] for bin, count in bin_counts.items()}
    log(f"Bin counts: {bin_counts}")

    undersamp_limit = min(bin_counts.values())

    if mode == 'none':
        pass
    elif mode == 'over':
        for bin in bins:
            n_under = oversamp_limit - bin_counts[bin]
            if n_under <= 0:
                # Undersample when bin count is over the limit
                new_samples = df[df['agebin'] == bin].sample(oversamp_limit, replace=False)
                df = df[df['agebin'] != bin]
                df = pd.concat([df, new_samples], axis=0)
            else:
                # Oversample as usual
                new_samples = df[df['agebin'] == bin].sample(n_under, replace=True)
                df = pd.concat([df, new_samples], axis=0)
    elif mode == 'under':
        for bin in bins:
            n_over = bin_counts[bin] - undersamp_limit
            assert n_over >= 0
            new_samples = df[df['agebin'] == bin].sample(undersamp_limit, replace=False)
            df = df[df['agebin'] != bin]
            df = pd.concat([df, new_samples], axis=0)
    elif mode == 'scale-up' or mode == 'scale-down':
        if mode == 'scale-up':
            target_count = oversamp_limit * len(bins)
            replace = True
        else:
            target_count = undersamp_limit * len(bins)
            replace = False

        for bin in bins:
            count = int(bin_ratios[bin] * target_count)
            new_samples = df[df['agebin'] == bin].sample(count, replace=replace)
            df = df[df['agebin'] != bin]
            df = pd.concat([df, new_samples], axis=0)
    else:
        raise Exception(f"Invalid sampling mode: {mode}")

    log(f"Number of samples in final training dataset: {df.shape[0]}")
    return df

def setup_model(arch, device):
    if arch == 'resnet18':
        model = models.resnet18(num_classes=1)
        # Set the number of input channels to 130
        model.conv1 = nn.Conv2d(130, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif arch == 'vgg8':
        model = VGG8(in_channels=130, num_classes=1)
    elif arch == 'sfcn':
        model = SFCN(in_channels=1, num_classes=1)
    elif arch == 'resnet18-20s':
        inner_model = models.resnet18(num_classes=1)
        inner_model.conv1 = nn.Conv2d(20, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model = SlidingWindow(inner_model, in_channels=130, window_size=20)
    elif arch == 'vgg8-20s':
        inner_model = VGG8(in_channels=20, num_classes=1)
        model = SlidingWindow(inner_model, in_channels=130, window_size=20)
    else:
        raise Exception(f"Invalid arch: {arch}")
    model.double()
    model.to(device)
    return model

def train(model, arch, optimizer, train_loader, device):
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
        if arch == 'sfcn':
            images = images.unsqueeze(1)
        age_preds = model(images).view(-1)
        loss = weighted_l1_loss(age_preds, ages, weights)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if batch_idx % 10 == 0:
            log(f"Batch {batch_idx} loss {loss} mean loss {np.mean(losses)}")
    
    return np.mean(losses)

def validate(model, arch, val_loader, device):
    model.eval()

    losses = []
    all_preds = []

    with torch.no_grad():
        for (images, ages, _) in val_loader:
            # When batch_size=1 DataLoader doesn't convert the data to Tensors
            if not torch.is_tensor(images):
                images = torch.tensor(images).unsqueeze(0)
            if not torch.is_tensor(ages):
                ages = torch.tensor(ages).unsqueeze(0)
            images, ages = images.to(device), ages.to(device)
            if arch == 'sfcn':
                images = images.unsqueeze(1)
            age_preds = model(images).view(-1)
            loss = F.l1_loss(age_preds, ages, reduction='mean')

            losses.append(loss.item())
            all_preds.extend(age_preds)
    
    return np.mean(losses), torch.stack(all_preds).cpu().numpy()

def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    opts = parse_options()

    log(f"Starting at {START_TIME}")

    checkpoint_dir = os.path.join(ROOT_DIR, "checkpoints", opts.eval or opts.job_id)
    results_dir = os.path.join(ROOT_DIR, "results", opts.job_id)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    log("Setting up dataset")

    split_fnames = glob.glob(f"{SPLITS_DIR}/nfold_imglist_all_nfold_*.list")
    assert len(split_fnames) == 5

    df = load_samples(split_fnames, opts.bin_width, opts.min_bin_count, opts.max_samples)
    _train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['agebin'])
    train_df = resample(_train_df, mode=opts.sample, oversamp_limit=opts.oversamp_limit)

    if opts.print_bin_counts:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            log(df['agebin'].value_counts())
            log(_train_df['agebin'].value_counts())
            log(train_df['agebin'].value_counts())
            log(val_df['agebin'].value_counts())
        sys.exit(0)

    train_dataset = AgePredictionDataset(train_df, reweight=opts.reweight, lds=opts.lds, lds_kernel=opts.lds_kernel, lds_ks=opts.lds_ks, lds_sigma=opts.lds_sigma)
    val_dataset = AgePredictionDataset(val_df)

    train_loader = data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=1)
    
    ## Dump configuration info to a JSON file
    
    with open(f"{results_dir}/config.json", 'w+') as f:
        cfg = vars(opts)
        cfg['start_time'] = START_TIME
        cfg['train_counts'] = json.loads(train_df['agebin'].value_counts().to_json())
        cfg['val_counts'] = json.loads(val_df['agebin'].value_counts().to_json())
        json.dump(cfg, f, sort_keys=True, indent=4)

    log("Setting up model")

    device = torch.device('cpu' if opts.cpu else 'cuda')
    model = setup_model(opts.arch, device)
    optimizer = optim.Adam(model.parameters(), lr=opts.initial_lr, weight_decay=opts.weight_decay)
    scheduler = StepLR(optimizer, step_size=opts.step_size, gamma=opts.gamma)

    ## Training and evaluation

    if opts.eval is None:
        log("Starting training process")

        best_epoch = None
        best_val_loss = np.inf
        train_losses = []
        val_losses = []

        for epoch in range(opts.n_epochs):
            log(f"Epoch {epoch}/{opts.n_epochs}")
            train_loss = train(model, opts.arch, optimizer, train_loader, device)
            log(f"Mean training loss: {train_loss}")
            val_loss, _ = validate(model, opts.arch, val_loader, device)
            log(f"Mean validation loss: {val_loss}")
            scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if val_loss < best_val_loss:
                torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pth")

                best_epoch = epoch
                best_val_loss = val_loss
        
        log(f"Best model had validation loss of {best_val_loss}, occurred at epoch {best_epoch}")
    
    log("Evaluating best model on val dataset")
   
    checkpoint = torch.load(f"{checkpoint_dir}/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint)
    
    _, best_val_preds = validate(model, opts.arch, val_dataset, device)
    
    ## Save results so we can plot them later

    log("Saving results")
    
    if opts.eval is None:
        np.savetxt(f"{results_dir}/train_losses_over_time.txt", train_losses)
        np.savetxt(f"{results_dir}/val_losses_over_time.txt", val_losses)

    val_df_with_preds = val_df.copy()
    val_df_with_preds['age_pred'] = best_val_preds
    val_df_with_preds.to_csv(f"{results_dir}/best_model_val_preds.csv", index=False)

if __name__ == '__main__':
    main()
