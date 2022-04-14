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
from .FiANet3D import fusNet
from .glt import GlobalLocalBrainAge
from .RelationNet import CompaireLearning
from .resnet import resnet18
from .sfcn import SFCN
from .vgg import VGG8

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SPLITS_DIR = os.path.join(ROOT_DIR, "folderlist")
START_TIME = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

print = logging.info

def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('arch', type=str, choices=['resnet18', 'vgg8', 'sfcn', 'glt', 'relnet', 'fianet'])
    parser.add_argument('--job-id', type=str, required=True, help='SLURM job ID')

    parser.add_argument('--batch-size', type=int, default=5, help='batch size')
    parser.add_argument('--n-epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--initial-lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--step-size', type=int, default=10, help='learning rate decay period')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor')
    parser.add_argument('--weight-decay', type=float, default=1e-6, help='weight decay')

    parser.add_argument('--splits-dir', type=str, default='default', help='name of splits directory')
    parser.add_argument('--sample', type=str, choices=['none', 'over', 'under', 'scale-up', 'scale-down'], default='none', help='sampling strategy')
    parser.add_argument('--reweight', type=str, choices=['none', 'inv', 'sqrt_inv'], default='none', help='reweighting strategy')

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

    args = parser.parse_args()
    assert (args.sample == 'none') or (args.reweight == 'none'), "--sample is incompatible with --reweight"
    assert (not args.from_checkpoint) or (args.n_epochs < 30), "Must specify --n-epochs alongside --from-checkpoint"

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

def setup_model(arch, device, checkpoint_file=None, fds=None):
    if arch == 'resnet18':
        model = resnet18(fds=fds)
        # Set the number of input channels to 130
        model.conv1 = nn.Conv2d(130, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif arch == 'vgg8':
        model = VGG8(in_channels=130, num_classes=1, fds=fds)
    elif arch == 'sfcn':
        model = SFCN(in_channels=1, num_classes=1, fds=fds)
    elif arch == 'glt':
        model = GlobalLocalBrainAge(inplace=130, num_classes=1, fds=fds)
    elif arch == 'relnet':
        model = CompaireLearning(in_dim=1)
        model.uses_fds = False
    elif arch == 'fianet':
        # TODO: add FDS support
        model = fusNet(inplace=1, num_classes=1)
        model.uses_fds = False
    else:
        raise Exception(f"Invalid arch: {arch}")
    if arch != 'fianet':
        model.double()
    model.to(device)
    
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint)

    return model

def train(model, arch, optimizer, train_loader, device, epoch):
    def weighted_l1_loss(inputs, targets, weights):
        inputs = inputs.view(-1)
        loss = F.l1_loss(inputs, targets, reduction='none')
        loss *= weights.expand_as(loss)
        loss = torch.mean(loss)
        return loss

    model.train()

    losses = []
    encodings = []
    targets = []

    is_3d = arch in ('sfcn', 'relnet', 'fianet')

    for batch_idx, inst in enumerate(train_loader):
        if arch == 'fianet':
            images, ravens_images, ages, weights = inst
            images, ravens_images, ages, weights = images.to(device), ravens_images.to(device), ages.to(device), weights.to(device)
        else:
            images, ages, weights = inst
            images, ages, weights = images.to(device), ages.to(device), weights.to(device)
        if is_3d:
            images = images.unsqueeze(1) # Add a dimension for the number of channels
        if arch == 'fianet':
            ravens_images = ravens_images.unsqueeze(1) # Add a dimension for the number of channels

        optimizer.zero_grad()

        if arch == 'glt':
            if model.uses_fds:
                age_bins = torch.floor(ages)
                age_preds_lst, batch_encodings_lst = model(images, targets=age_bins, epoch=epoch)

                for batch_encodings in batch_encodings_lst:
                    encodings.extend(batch_encodings.detach().cpu().numpy())
                    targets.extend(age_bins.cpu().numpy())
            else:
                age_preds_lst = model(images)
            loss = torch.sum(torch.stack(
                [weighted_l1_loss(age_preds, ages, weights) for age_preds in age_preds_lst]
            ))
        elif arch == 'relnet':
            batch_len = images.shape[0]
            if batch_len % 2 != 0:
                print(f"Batch {batch_idx} has odd size of {batch_len}, skipping")
                continue

            im1, im2 = torch.chunk(images, 2, dim=0)
            a1, a2 = torch.chunk(ages, 2, dim=0)
            w1, w2 = torch.chunk(weights, 2, dim=0)

            outlist = model(im1, im2)
            loss = 0.

            true_sum = a1 + a2
            loss += weighted_l1_loss(outlist[0].squeeze(1), true_sum, (w1 + w2)/2)

            true_diff = a1 - a2
            loss += weighted_l1_loss(outlist[1].squeeze(1), true_diff, (w1 + w2)/2)

            true_max = torch.max(a1, a2)
            loss += weighted_l1_loss(outlist[2].squeeze(1), true_max, (w1 + w2)/2)

            true_min = torch.min(a1, a2)
            loss += weighted_l1_loss(outlist[3].squeeze(1), true_min, (w1 + w2)/2)
        elif arch == 'fianet':
            if model.uses_fds:
                # TODO: add FDS support
                pass
            else:
                age_preds_1, age_preds_2, age_preds_combined = model(images, ravens_images)
            loss = weighted_l1_loss(age_preds_1, ages, weights)
            loss += weighted_l1_loss(age_preds_2, ages, weights)
            loss += weighted_l1_loss(age_preds_combined, ages, weights)
        else:
            if model.uses_fds:
                age_bins = torch.floor(ages)
                age_preds, batch_encodings = model(images, targets=age_bins, epoch=epoch)
                encodings.extend(batch_encodings.detach().cpu().numpy())
                targets.extend(age_bins.cpu().numpy())
            else:
                age_preds = model(images)
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
    def stack2np(tensor):
        return torch.stack(tensor).cpu().numpy()

    model.eval()

    losses = []

    if arch == 'relnet':
        sum_preds = []
        max_preds = []
        min_preds = []
    else:
        all_preds = []
    
    is_3d = arch in ('sfcn', 'relnet', 'fianet')

    with torch.no_grad():
        for inst in val_loader:
            if arch == 'fianet':
                images, ravens_images, ages, _ = inst
                images, ravens_images, ages = images.to(device), ravens_images.to(device), ages.to(device)
            else:
                images, ages, _ = inst
                images, ages = images.to(device), ages.to(device)
            if is_3d:
                images = images.unsqueeze(1) # Add a dimension for the number of channels
            if arch == 'fianet':
                ravens_images = ravens_images.unsqueeze(1) # Add a dimension for the number of channels
            
            if arch == 'relnet':
                outlist = model(images, images)
                pred_sum, pred_max, pred_min = outlist[0].squeeze(1), outlist[2].squeeze(1), outlist[3].squeeze(1)

                sum_loss = F.l1_loss(pred_sum / 2, ages, reduction='mean')
                max_loss = F.l1_loss(pred_max, ages, reduction='mean')
                min_loss = F.l1_loss(pred_min, ages, reduction='mean')
                loss = (sum_loss + max_loss + min_loss) / 3
                losses.append(loss.item())

                sum_preds.extend((pred_sum / 2).view(-1))
                max_preds.extend(pred_max.view(-1))
                min_preds.extend(pred_min.view(-1))
            else:
                if arch == 'fianet':
                    age_preds = model(images, ravens_images)
                else:
                    age_preds = model(images)
                age_preds = age_preds.view(-1)
                loss = F.l1_loss(age_preds, ages, reduction='mean')
                losses.append(loss.item())
                all_preds.extend(age_preds)
    
    if arch == 'relnet':
        return np.mean(losses), {
            'sum': stack2np(sum_preds),
            'max': stack2np(max_preds),
            'min': stack2np(min_preds)
        }
    
    return np.mean(losses), stack2np(all_preds)

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

    splits_dir = os.path.join(ROOT_DIR, "splits", opts.splits_dir)
    train_df = pd.read_csv(os.path.join(splits_dir, "train" if opts.sample == 'none' else f"train_{opts.sample}"))
    val_df = pd.read_csv(os.path.join(splits_dir, "val"))

    lds = {
        'kernel': opts.lds_kernel,
        'ks': opts.lds_ks,
        'sigma': opts.lds_sigma
    } if opts.lds else None
    train_dataset = AgePredictionDataset(train_df, reweight=opts.reweight, lds=lds, fianet=(opts.arch == 'fianet'))
    val_dataset = AgePredictionDataset(val_df, fianet=(opts.arch == 'fianet'))

    train_loader = data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=opts.batch_size)
    
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
    
    print("Evaluating best model on val dataset")
   
    checkpoint = torch.load(f"{checkpoint_dir}/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint)
    
    _, best_val_preds = validate(model, opts.arch, val_loader, device)
    
    ## Save results so we can plot them later

    print("Saving results")
    
    if opts.eval is None:
        np.savetxt(f"{results_dir}/train_losses_over_time.txt", train_losses)
        np.savetxt(f"{results_dir}/val_losses_over_time.txt", val_losses)

    val_df_with_preds = val_df.copy()
    if opts.arch == 'relnet':
        val_df_with_preds['age_pred_sum'] = best_val_preds['sum']
        val_df_with_preds['age_pred_max'] = best_val_preds['max']
        val_df_with_preds['age_pred_min'] = best_val_preds['min']
    else:
        val_df_with_preds['age_pred'] = best_val_preds
    val_df_with_preds.to_csv(f"{results_dir}/best_model_val_preds.csv", index=False)

if __name__ == '__main__':
    main()
