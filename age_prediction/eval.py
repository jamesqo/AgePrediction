import torch
from torch.utils import data

from .dataset import AgePredictionDataset
from .train import setup_model, validate

def predict_ages(df, arch='resnet18', device='cpu'):
    model_path = f'/neuro/labs/grantlab/MRI_Predict_Age/james.ko/AgePredictionModels/best_{arch}_model_trainedon_0_100.pth'
    checkpoint = torch.load(model_path, map_location=device)
    model = setup_model(arch, device)
    model.load_state_dict(checkpoint)

    dataset = AgePredictionDataset(df)
    dataloader = data.DataLoader(dataset)
    _, age_preds = validate(model, dataloader, device)

    return age_preds
