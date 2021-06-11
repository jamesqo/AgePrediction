import argparse
from datetime import datetime
import glob
import os
import random

import matplotlib.pyplot as plt
import nibabel
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from vgg import VGG8

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
LOG_FILE = f"{SCRIPT_DIR}/logs/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def log(message):
    with open(LOG_FILE, 'a+') as log_file:
        log_file.write(f"{message}\n")

class AgePredictionDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = nibabel.load(row['path']).get_fdata()
        image = image[54:184,25:195,12:132] # Crop out zeroes
        image /= np.percentile(image, 95) # Normalize intensity
        age = row['age']
        return (image, age)

    def __len__(self):
        return len(self.df)

def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('arch', type=str)

    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--n-epochs', type=int, default=30)
    parser.add_argument('--initial-lr', type=float, default=1e-3)
    parser.add_argument('--step-size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--weight-decay', type=float, default=1e-6)

    parser.add_argument('--max-samples', type=int)

    args = parser.parse_args()
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

def train(model, optimizer, criterion, train_loader):
    model.train()

    losses = []

    for batch_idx, (images, ages) in enumerate(train_loader):
        images, ages = images.cuda(), ages.cuda()
        optimizer.zero_grad()
        age_preds = model(images).view(-1)
        loss = criterion(age_preds, ages)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if batch_idx % 10 == 0:
            log(f"Batch {batch_idx} loss {loss} mean loss {np.mean(losses)}")
    
    return np.mean(losses)

def validate(model, criterion, val_loader):
    model.eval()

    losses = []

    with torch.no_grad():
        for (images, ages) in val_loader:
            images, ages = images.cuda(), ages.cuda()
            age_preds = model(images).view(-1)
            loss = criterion(age_preds, ages)

            losses.append(loss.item())
    
    return np.mean(losses)

def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    opts = parse_options()
    
    os.makedirs(f"{SCRIPT_DIR}/checkpoints", exist_ok=True)
    os.makedirs(f"{SCRIPT_DIR}/figures", exist_ok=True)

    split_fnames = glob.glob(f"{SCRIPT_DIR}/folderlist/nfold_imglist_all_nfold_*.list")
    assert len(split_fnames) == 5

    log("Setting up model")

    if opts.arch == 'resnet18':
        model = models.resnet18(num_classes=1)
        # Set the number of input channels to 240
        model.conv1 = nn.Conv2d(240, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif opts.arch == 'vgg8':
        model = VGG8(in_channels=240, num_classes=1)
    else:
        raise Exception(f"Invalid arch: {opts.arch}")
    model.double()
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=opts.initial_lr, weight_decay=opts.weight_decay)
    scheduler = StepLR(optimizer, step_size=opts.step_size, gamma=opts.gamma)
    criterion = nn.L1Loss(reduction='mean')

    log("Setting up dataset")

    df = load_samples(split_fnames, opts.max_samples)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['agebin'])
    train_dataset = AgePredictionDataset(train_df)
    val_dataset = AgePredictionDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opts.batch_size)

    ## Training process

    log("Starting training process")

    best_epoch = None
    best_val_loss = np.inf
    train_losses = []
    val_losses = []

    for epoch in range(opts.n_epochs):
        log(f"Epoch {epoch}/{opts.n_epochs}")
        train_loss = train(model, optimizer, criterion, train_loader)
        log(f"Mean training loss: {train_loss}")
        val_loss = validate(model, criterion, val_loader)
        log(f"Mean validation loss: {val_loss}")
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), f"{SCRIPT_DIR}/checkpoints/best_model.pth")

            best_epoch = epoch
            best_val_loss = val_loss
    
    log(f"Best model had validation loss of {best_val_loss}, occurred at epoch {best_epoch}")

    ## Plotting

    log("Generating plots")

    epochs = range(opts.n_epochs)
    plt.plot(epochs, train_losses)
    plt.plot(epochs, val_losses)
    plt.legend(["training loss", "validation loss"])
    plt.savefig(f"{SCRIPT_DIR}/figures/train_and_val_loss_vs_epoch.png")
    plt.clf()

if __name__ == '__main__':
    main()
