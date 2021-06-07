#!/bin/bash

#SBATCH --partition=bch-gpu
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:Titan_RTX:1
#SBATCH --output=logs/slurm.out

module load anaconda3
source activate james
pip install --user -r requirements.txt
python main.py
