#!/bin/bash

#SBATCH --partition=fnndsc-gpu
#SBATCH --account=fnndsc
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:Titan_RTX:1
#SBATCH --output=logs/slurm-%j.out

module load anaconda3
source activate james
pip install --user -r requirements.txt
python -m age_prediction.train --job-id $SLURM_JOB_ID "$@"
