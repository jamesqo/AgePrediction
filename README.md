# AgePrediction

## Installation

Clone this Git repository on E2:

```
ssh chXXXXXX@e2.tch.harvard.edu
git clone git@github.com:jamesqo/AgePrediction.git
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
Parameters:
- architecture: CNN architecture to use. 'resnet18' or 'vgg8'
- sampling_mode: The sampling strategy that was used during model training. 'none', 'over', 'under', 'scale-up', or 'scale-down'
- device: Where the model should be evaluated. 'cpu' or 'gpu'
"""
predict_ages(dataframe, architecture=..., sampling_mode=..., device=...)
```
