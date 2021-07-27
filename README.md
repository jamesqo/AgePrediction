# AgePrediction

## Installation

Clone this Git repository on E2 and switch to the `release` branch:

```
ssh chXXXXXX@e2.tch.harvard.edu
git clone git@github.com:jamesqo/AgePrediction.git
git checkout release
```

Create a virtual environment for your project:

```
cd /path/to/my/project
conda create -n my_env_name python=3.6
```

Install the module:

```
pip install -e /path/to/AgePrediction
```

## Usage

```py
from age_prediction.eval import predict_ages

"""
This should be a Pandas DataFrame with two columns:
- 'path' containing the path to the MRI image
- 'age' containing the subject's true age
"""
dataframe = ...

"""
Optional parameters:
- architecture: CNN architecture to use. 'resnet18' or 'vgg8'
- age_range: The age range of the dataset that the model was trained on. '0-100'
- sampling_mode: The sampling strategy that was used during model training. 'none', 'over', 'under', 'scale-up', or 'scale-down'
- weighting: The reweighting strategy that was used during model training. 'none', 'inv', or 'sqrt_inv'
- lds: Whether or not label distribution smoothing (LDS) was used during model training. True or False
- device: Where the model should be evaluated. 'cpu' or 'gpu'
"""
predict_ages(dataframe,
             architecture=...,
             age_range=...,
             sampling_mode=...,
             weighting=...,
             lds=...,
             device=...)
```
