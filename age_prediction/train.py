import argparse
from datetime import datetime
import glob
import json
import logging
import os
import random
import re
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

from .dataset import AgePredictionDataset
from .resnet import resnet18
from .sfcn import SFCN
from .vgg import VGG8

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SPLITS_DIR = os.path.join(ROOT_DIR, "folderlist")
START_TIME = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
LOG_FILE = ''

print = logging.info

def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('arch', type=str, choices=['resnet18', 'vgg8', 'sfcn'])
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
    parser.add_argument('--undersamp-limit', type=int, default=18, help='minimum number of samples each age bin will be deflated to when undersampling (bins with fewer samples will be oversampled)')
    parser.add_argument('--oversamp-limit', type=int, default=126, help='maximum number of samples each age bin will be inflated to when oversampling (bins with more samples will be undersampled)')

    parser.add_argument('--lds', action='store_true', default=False, help='whether to enable LDS')
    parser.add_argument('--lds_kernel', type=str, default='gaussian',
                        choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
    parser.add_argument('--lds_ks', type=int, default=5, help='LDS kernel size: should be odd number')
    parser.add_argument('--lds_sigma', type=float, default=2, help='LDS gaussian/laplace kernel sigma')

    parser.add_argument('--fds', action='store_true', default=False, help='whether to enable FDS')
    parser.add_argument('--fds_kernel', type=str, default='gaussian', choices=['gaussian', 'triang', 'laplace'], help='FDS kernel type')
    parser.add_argument('--fds_ks', type=int, default=5, help='FDS kernel size: should be odd number')
    parser.add_argument('--fds_sigma', type=float, default=2, help='FDS gaussian/laplace kernel sigma')
    parser.add_argument('--start_update', type=int, default=0, help='which epoch to start FDS updating')
    parser.add_argument('--start_smooth', type=int, default=1, help='which epoch to start using FDS to smooth features')
    parser.add_argument('--bucket_num', type=int, default=100, help='maximum bucket considered for FDS')
    parser.add_argument('--bucket_start', type=int, default=0, help='minimum (starting) bucket for FDS')
    parser.add_argument('--fds_momentum', type=float, default=0.9, help='FDS momentum')

    parser.add_argument('--from-checkpoint', type=str, help='continue training from an existing checkpoint')

    ## For testing purposes only
    parser.add_argument('--cpu', action='store_true', help='use CPU instead of GPU')
    parser.add_argument('--eval', type=str, help='evaluate a pretrained model')
    parser.add_argument('--max-samples', type=int, help='limit the number of samples used for training/validation')
    parser.add_argument('--save-df', action='store_true', help='save full dataframe to results dir and exit')
    parser.add_argument('--print-bin-counts', action='store_true', help='print age bin counts and exit')
    parser.add_argument('--val-size', type=float, default=0.2, help='size of validation set used for training')

    args = parser.parse_args()
    assert (args.sample == 'none') or (args.reweight == 'none'), "--sample is incompatible with --reweight"
    assert (not args.from_checkpoint) or (args.n_epochs < 30), "Must specify --n-epochs alongside --from-checkpoint"

    global LOG_FILE
    LOG_FILE = os.path.join(ROOT_DIR, "logs", f"{args.job_id}.log")
    try:
        os.remove(LOG_FILE)
    except FileNotFoundError:
        pass
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, filename=LOG_FILE, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print('='*20)
    return args

def load_samples(split_fnames, bin_width=1, min_bin_count=10, max_samples=None):
    def correct_path(path):
        path = path.replace('/ABIDE/', '/ABIDE_I/')
        path = path.replace('/NIH-PD/', '/NIH_PD/')
        return path
    
    def extract_dataset(path):
        name = re.match(r'/neuro/labs/grantlab/research/MRI_Predict_Age/([^/]*)', path)[1]
        return 'MGHBCH' if name in ('MGH', 'BCH') else name

    schema = {'id': str, 'age': float, 'sex': str, 'path': str}
    dfs = []
    for fname in split_fnames:
        split_num = int(fname[:-len('.list')].split('_')[-1])

        df = pd.read_csv(fname, sep=' ', header=None, names=['id', 'age', 'sex', 'path'], dtype=schema)
        df['path'] = df['path'].apply(correct_path)
        df['agebin'] = df['age'] // bin_width
        df['split'] = split_num
        df['dataset'] = df['path'].apply(extract_dataset)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, axis=0)
    
    if max_samples is not None:
        combined_df = combined_df.sample(max_samples)

    if min_bin_count is not None:
        bin_counts = combined_df['agebin'].value_counts()
        bins_below_cutoff = [bin for bin in bin_counts.keys() if bin_counts[bin] < min_bin_count]
        combined_df = combined_df[~combined_df['agebin'].isin(bins_below_cutoff)]
    
    return combined_df

def resample(df, mode, undersamp_limit, oversamp_limit):
    print("Resampling training data")

    bins = sorted(set(df['agebin']))
    bin_counts = {bin: sum(df['agebin'] == bin) for bin in bins}
    bin_ratios = {bin: count / df.shape[0] for bin, count in bin_counts.items()}
    print(f"Bin counts: {bin_counts}")

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
            if n_over <= 0:
                # Oversample when bin count is under the limit
                new_samples = df[df['agebin'] == bin].sample(-n_over, replace=True)
                df = pd.concat([df, new_samples], axis=0)
            else:
                # Undersample as usual
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

    print(f"Number of images in final training dataset: {df.shape[0]}")
    return df

def setup_model(arch, device, checkpoint_file=None, fds=None):
    if arch == 'resnet18':
        model = resnet18(fds=fds)
        # Set the number of input channels to 130
        model.conv1 = nn.Conv2d(130, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif arch == 'vgg8':
        model = VGG8(in_channels=130, num_classes=1, fds=fds)
    elif arch == 'sfcn':
        model = SFCN(in_channels=1, num_classes=1, fds=fds)
    else:
        raise Exception(f"Invalid arch: {arch}")
    model.double()
    model.to(device)
    
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint)

    return model

def train(model, arch, optimizer, train_loader, device, epoch):
    def weighted_l1_loss(inputs, targets, weights):
        loss = F.l1_loss(inputs, targets, reduction='none')
        loss *= weights.expand_as(loss)
        loss = torch.mean(loss)
        return loss

    model.train()

    losses = []

    encodings = []
    targets = []

    for batch_idx, (images, ages, weights) in enumerate(train_loader):
        images, ages, weights = images.to(device), ages.to(device), weights.to(device)
        optimizer.zero_grad()
        if arch == 'sfcn':
            images = images.unsqueeze(1)
        if model.uses_fds:
            age_bins = torch.floor(ages)
            age_preds, batch_encodings, _ = model(images, targets=age_bins, epoch=epoch)
            encodings.extend(batch_encodings.detach().cpu().numpy())
            targets.extend(age_bins.cpu().numpy())
        else:
            age_preds, _ = model(images)
        age_preds = age_preds.view(-1)
        loss = weighted_l1_loss(age_preds, ages, weights)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx} loss {loss} mean loss {np.mean(losses)}")
        
    if model.uses_fds:
        encodings, targets = torch.from_numpy(np.vstack(encodings)).to(device), torch.from_numpy(np.hstack(targets)).to(device)
        model.fds.update_last_epoch_stats(epoch)
        model.fds.update_running_stats(encodings, targets, epoch)
    
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
            age_preds, _ = model(images)
            age_preds = age_preds.view(-1)
            loss = F.l1_loss(age_preds, ages, reduction='mean')

            losses.append(loss.item())
            all_preds.extend(age_preds)
    
    return np.mean(losses), torch.stack(all_preds).cpu().numpy()

def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    opts = parse_options()

    print(f"Starting at {START_TIME}")

    checkpoint_dir = os.path.join(ROOT_DIR, "checkpoints", opts.eval or opts.job_id)
    results_dir = os.path.join(ROOT_DIR, "results", opts.job_id)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("Setting up dataset")

    split_fnames = glob.glob(f"{SPLITS_DIR}/nfold_imglist_all_nfold_*.list")
    assert len(split_fnames) == 5

    if opts.save_df:
        opts.min_bin_count = None
    df = load_samples(split_fnames, opts.bin_width, opts.min_bin_count, opts.max_samples)
    if opts.save_df:
        df.to_csv(os.path.join(results_dir, "merged_df.csv"), index=False)
        sys.exit(0)

    _train_df, val_df = train_test_split(df, test_size=opts.val_size, stratify=df['agebin'])
    train_df = resample(_train_df, mode=opts.sample, undersamp_limit=opts.undersamp_limit, oversamp_limit=opts.oversamp_limit)

    if opts.print_bin_counts:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df['agebin'].value_counts())
            print(_train_df['agebin'].value_counts())
            print(train_df['agebin'].value_counts())
            print(val_df['agebin'].value_counts())
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

    print("Setting up model")

    device = torch.device('cpu' if opts.cpu else 'cuda')
    fds = None
    if opts.fds:
        fds = {
            'bucket_num': opts.bucket_num,
            'bucket_start': opts.bucket_start,
            'start_update': opts.start_update,
            'start_smooth': opts.start_smooth,
            'kernel': opts.fds_kernel,
            'ks': opts.fds_ks,
            'sigma': opts.fds_sigma,
            'momentum': opts.fds_momentum
        }
    model = setup_model(opts.arch, device, checkpoint_file=opts.from_checkpoint, fds=fds)
    optimizer = optim.Adam(model.parameters(), lr=opts.initial_lr, weight_decay=opts.weight_decay)
    scheduler = StepLR(optimizer, step_size=opts.step_size, gamma=opts.gamma)

    ## Training and evaluation

    if opts.eval is None:
        print("Starting training process")

        best_epoch = None
        best_val_loss = np.inf
        train_losses = []
        val_losses = []

        if opts.n_epochs == 0:
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pth")

        for epoch in range(opts.n_epochs):
            print(f"Epoch {epoch}/{opts.n_epochs}")
            train_loss = train(model, opts.arch, optimizer, train_loader, device, epoch)
            print(f"Mean training loss: {train_loss}")
            val_loss, _ = validate(model, opts.arch, val_loader, device)
            print(f"Mean validation loss: {val_loss}")
            scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            torch.save(model.state_dict(), f"{checkpoint_dir}/epoch{epoch}.pth")

            if val_loss < best_val_loss:
                torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pth")

                best_epoch = epoch
                best_val_loss = val_loss
        
        print(f"Best model had validation loss of {best_val_loss}, occurred at epoch {best_epoch}")
    
    ## Before we reload the model from the checkpoint file, save the features generated by FDS which can't(?)
    ## be loaded from the checkpoint file

    if opts.fds:
        assert model.uses_fds

        train_features_before_smoothing = []
        train_features_after_smoothing = []
        val_features = []

        model.train()
        seq_train_loader = data.DataLoader(train_dataset, batch_size=1)
        for images, ages, weights in seq_train_loader:
            images, ages, weights = images.to(device), ages.to(device), weights.to(device)
            if opts.arch == 'sfcn':
                images = images.unsqueeze(1)
            _, encodings, encodings_s = model(images, targets=ages, epoch=epoch)
            print(encodings.shape)
            print(encodings_s.shape)
            train_features_before_smoothing.append(encodings.detach().cpu().numpy().flatten()[:20])
            train_features_after_smoothing.append(encodings_s.detach().cpu().numpy().flatten()[:20])
            l2_distance = np.linalg.norm(encodings_s.detach().cpu().numpy().flatten() - encodings.detach().cpu().numpy().flatten())
            print(l2_distance)
        
        model.eval()
        for images, ages, weights in val_loader:
            images, ages, weights = images.to(device), ages.to(device), weights.to(device)
            if opts.arch == 'sfcn':
                images = images.unsqueeze(1)
            _, encodings = model(images)
            print(encodings.shape)
            val_features.append(encodings.detach().cpu().numpy().flatten()[:20])
        
        colnames = [f'feat{f}' for f in range(20)]
        train_features_before_smoothing = pd.DataFrame(train_features_before_smoothing, columns=colnames)
        train_features_after_smoothing = pd.DataFrame(train_features_after_smoothing, columns=colnames)
        val_features = pd.DataFrame(val_features, columns=colnames)

        '''
        train_features_before_smoothing = pd.concat([train_df, train_features_before_smoothing], axis=1)
        train_features_after_smoothing = pd.concat([train_df, train_features_after_smoothing], axis=1)
        val_features = pd.concat([val_df, val_features], axis=1)
        '''

        """
        for df in (train_features_before_smoothing, train_features_after_smoothing):
            df['id'] = train_df['id']
            df['age'] = train_df['age']
            df['sex'] = train_df['sex']
            df['path'] = train_df['path']
        
        val_features['id'] = val_df['id']
        val_features['age'] = val_df['age']
        val_features['sex'] = val_df['sex']
        val_features['path'] = val_df['path']
        """

        train_df.to_csv(f"{results_dir}/train_df.csv", index=False)
        val_df.to_csv(f"{results_dir}/val_df.csv", index=False)

        train_features_before_smoothing.to_csv(f"{results_dir}/train_features_before_smoothing.csv", index=False)
        train_features_after_smoothing.to_csv(f"{results_dir}/train_features_after_smoothing.csv", index=False)
        val_features.to_csv(f"{results_dir}/val_features.csv", index=False)
    
    print("Evaluating best model on val dataset")
   
    checkpoint = torch.load(f"{checkpoint_dir}/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint)
    
    _, best_val_preds = validate(model, opts.arch, val_dataset, device)
    
    ## Save results so we can plot them later

    print("Saving results")
    
    if opts.eval is None:
        np.savetxt(f"{results_dir}/train_losses_over_time.txt", train_losses)
        np.savetxt(f"{results_dir}/val_losses_over_time.txt", val_losses)

    val_df_with_preds = val_df.copy()
    val_df_with_preds['age_pred'] = best_val_preds
    val_df_with_preds.to_csv(f"{results_dir}/best_model_val_preds.csv", index=False)

if __name__ == '__main__':
    main()
