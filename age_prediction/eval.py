import os

import torch
from torch.utils import data

from .dataset import AgePredictionDataset
from .train import setup_model, validate

def predict_ages(df, architecture='resnet18', sampling_mode='none', device='cpu'):
    dirname = '/neuro/labs/grantlab/MRI_Predict_Age/james.ko/AgePredictionModels'
    model_path = f'{dirname}/arch_{architecture}_sample_{sampling_mode}_trainedon_0-100.pth'
    if not os.path.isfile(model_path):
        raise Exception(f"{model_path} doesn't exist")
    checkpoint = torch.load(model_path, map_location=device)
    model = setup_model(architecture, device)
    model.load_state_dict(checkpoint)

    dataset = AgePredictionDataset(df)
    dataloader = data.DataLoader(dataset)
    _, age_preds = validate(model, dataloader, device)

    return age_preds
