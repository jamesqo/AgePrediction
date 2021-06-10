import argparse
from datetime import datetime
import glob
import os
import random

import matplotlib.pyplot as plt
import nibabel
import numpy as np
import torch
from torch import nn
from torch import optim
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
    def __init__(self, split_fnames, max_samples=None):
        rows = []

        for fname in split_fnames:
            split_num = int(fname[:-len('.list')].split('_')[-1])
            with open(fname, 'r') as file:
                lines = file.read().splitlines()
                for line in lines:
                    id, age, sex, path = line.split(' ')
                    path = path.replace('/ABIDE/', '/ABIDE_I/')
                    path = path.replace('/NIH-PD/', '/NIH_PD/')
                    rows.append((id, float(age), sex, path))

        if max_samples is not None:
            rows = random.sample(rows, max_samples)

        self.rows = rows

    def __getitem__(self, idx):
        id, age, sex, path = self.rows[idx]
        image = nibabel.load(path).get_fdata()
        image /= np.percentile(image, 95) # Normalize intensity
        return (image, age)

    def __len__(self):
        return len(self.rows)

def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('arch', type=str)

    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--n-epochs', type=int, default=30)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-6)

    parser.add_argument('--max-samples', type=int)

    args = parser.parse_args()
    for arg in vars(args):
        log(f"{arg}: {getattr(args, arg)}")
    log('='*20)
    return args

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

    optimizer = optim.Adam(model.parameters(), lr=opts.learning_rate, weight_decay=opts.weight_decay)
    criterion = nn.L1Loss(reduction='mean')

    log("Setting up dataset")

    dataset = AgePredictionDataset(split_fnames, opts.max_samples)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])

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
