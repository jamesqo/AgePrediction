# AgePrediction

## Installation

Clone this Git repository on E2 and switch to the `release` branch:

```
ssh chXXXXXX@e2.tch.harvard.edu
git clone git@github.com:jamesqo/AgePrediction.git
git checkout release
```

Create and activate a virtual environment for your project:

```
cd /path/to/my/project
module load anaconda3
conda create -n <env_name> python=3.6
conda activate <env_name>
```

Install the module into your virtual environment:

```
pip install -e /path/to/AgePrediction
```

After you've finished writing your script, create a file called `slurm` in your project directory with the following contents:

```bash
#!/bin/bash

#SBATCH --partition=bch-gpu
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:Titan_RTX:1

module load anaconda3
source activate <env_name>
python my_script.py "$@"
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

Returns: a NumPy array containing the predicted ages
"""
predict_ages(dataframe,
             architecture=...,
             age_range=...,
             sampling_mode=...,
             weighting=...,
             lds=...,
             device=...)
```

## SLURM Cheatsheet

To submit a new job:

```
sbatch slurm <program args>
```

To view a list of running jobs:

```
squeue -u $USER
```

To cancel a job:

```
scancel <job_id>
```
