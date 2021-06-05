import argparse
import glob
import os

import matplotlib.pyplot as plt
import nibabel
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

class AgePredictionDataset(Dataset):
    def __init__(self, split_fnames):
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

        self.rows = rows

    def __getitem__(self, idx):
        id, age, sex, path = self.rows[idx]
        image = nibabel.load(path).get_fdata()
        return (image, age)

    def __len__(self):
        return len(self.rows)

def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--n-epochs', type=int, default=30)

    parser.add_argument('--learning-rate', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    return parser.parse_args()

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
            print(f"Batch {batch_idx} loss {loss} mean loss {np.mean(losses)}")
    
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
    opts = parse_options()
    
    os.makedirs(f"{SCRIPT_DIR}/checkpoints", exist_ok=True)
    os.makedirs(f"{SCRIPT_DIR}/figures", exist_ok=True)

    split_fnames = glob.glob(f"{SCRIPT_DIR}/folderlist/nfold_imglist_all_nfold_*.list")
    assert len(split_fnames) == 5

    model = models.resnet18(num_classes=1)
    # Set the number of input channels to 240
    model.conv1 = nn.Conv2d(240, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.double()
    model.cuda()
    optimizer = optim.SGD(
        model.parameters(),
        lr=opts.learning_rate,
        momentum=opts.momentum,
        weight_decay=opts.weight_decay)
    criterion = nn.L1Loss(reduction='mean')

    train_dataset = AgePredictionDataset(split_fnames[:4])
    val_dataset = AgePredictionDataset(split_fnames[4:])

    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=opts.batch_size)

    ## Training process

    best_epoch = None
    best_val_loss = np.inf
    train_losses = []
    val_losses = []

    for epoch in range(opts.n_epochs):
        print(f"Epoch {epoch}/{opts.n_epochs}")
        train_loss = train(model, optimizer, criterion, train_loader)
        print(f"Mean training loss: {train_loss}")
        val_loss = validate(model, criterion, val_loader)
        print(f"Mean validation loss: {val_loss}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), f"{SCRIPT_DIR}/checkpoints/best_model.pth")

            best_epoch = epoch
            best_val_loss = val_loss
    
    print(f"Best model had validation loss of {best_val_loss}, occurred at epoch {best_epoch}")

    ## Plotting

    epochs = range(opts.n_epochs)
    plt.plot(epochs, train_losses)
    plt.plot(epochs, val_losses)
    plt.legend(["training loss", "validation loss"])
    plt.savefig(f"{SCRIPT_DIR}/figures/train_and_val_loss_vs_epoch.png")
    plt.clf()

if __name__ == '__main__':
    main()
