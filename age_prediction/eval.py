import os

import pandas as pd
import torch
from torch.utils import data

from .dataset import AgePredictionDataset
from .train import setup_model

def evaluate(model, arch, dataloader, device):
    model.eval()

    all_preds = []

    with torch.no_grad():
        for images in dataloader:
            # When batch_size=1 DataLoader doesn't convert the data to Tensors
            if not torch.is_tensor(images):
                images = torch.tensor(images).unsqueeze(0)
            images = images.to(device)
            if arch == 'sfcn':
                images = images.unsqueeze(1)
            age_preds = model(images).view(-1)

            all_preds.extend(age_preds)
    
    return torch.stack(all_preds).cpu().numpy()

def predict_ages(filenames,
                 architecture='resnet18',
                 age_range='0-100',
                 sampling_mode='none',
                 weighting='none',
                 lds=False,
                 device='cuda'):
    if device == 'cuda' and not torch.cuda.is_available():
        raise Exception("You need to be running this code from SLURM. See the README for instructions")

    dirname = '/neuro/labs/grantlab/research/MRI_Predict_Age/james.ko/AgePredictionModels'
    model_path = f'{dirname}/arch_{architecture}__agerange_{age_range}__sample_{sampling_mode}__reweight_{weighting}__lds_{lds}.pth'
    if not os.path.isfile(model_path):
        raise Exception(f"{model_path} doesn't exist")
    checkpoint = torch.load(model_path, map_location=device)
    model = setup_model(architecture, device)
    model.load_state_dict(checkpoint)

    df = pd.DataFrame({'path': filenames})
    dataset = AgePredictionDataset(df, labeled=False)
    dataloader = data.DataLoader(dataset)
    age_preds = evaluate(model, architecture, dataloader, device)

    return pd.DataFrame({'path': filenames, 'age_pred': age_preds})
