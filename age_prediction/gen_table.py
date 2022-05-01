import re
import sys
from collections import defaultdict

import os
import json

def display_arch(arch):
    return {
        'resnet18': "ResNet18",
        'vgg8': "VGG",
        'sfcn': "SFCN",
        'glt': "GL-Transformer",
        '3dt': "mSFCN+Transformer"
    }[arch]

def display_strat(strat):
    return {
        'baseline': 'Baseline',
        'under': 'Undersampling',
        'scale-down': 'Scaling down',
        'over': 'Oversampling',
        'scale-up': 'Scaling up',
        'inv': 'Inv',
        'lds+inv': 'Inv + LDS',
        'fds+inv': 'Inv + FDS',
        'lds+fds+inv': 'Inv + LDS + FDS',
        'sqrt_inv': 'SqInv',
        'lds+sqrt_inv': 'SqInv + LDS',
        'fds+sqrt_inv': 'SqInv + FDS',
        'lds+fds+sqrt_inv': 'SqInv + LDS + FDS'
    }[strat]

def to3f(fl):
    return f"{fl:.3f}"

def main():
    arch = sys.argv[1]
    mode = sys.argv[2]

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(f'{ROOT_DIR}/results/all_losses.json', 'r', encoding='utf8') as f:
        results = json.load(f)
    
    if mode == 'rs':
        strats = ['baseline', 'under', 'scale-down', 'over', 'scale-up']
    elif mode == 'rw':
        strats = ['baseline', 'inv', 'lds+inv', 'fds+inv', 'lds+fds+inv', 'sqrt_inv', 'lds+sqrt_inv', 'fds+sqrt_inv', 'lds+fds+sqrt_inv']
    
    ## Write header
    print(f"\t&\\multirow{{3}}{{*}}{{{display_arch(arch)}}} & ", end='')

    ## Write MAEs
    maes = [to3f(results[arch][strat]['mae']['overall_mae']) for strat in strats]
    print(f"MAE & {' & '.join(maes)} \\\\")

    ## Write entropies
    entropies = [to3f(results[arch][strat]['mae']['entropy']) for strat in strats]
    print(f"\t& &  $\\mathcal{{S}}$ & {' & '.join(entropies)} \\\\")

    ## Write criterions
    criterions = [to3f(results[arch][strat]['mae']['criterion']) for strat in strats]
    print(f"\t& &  MAE$\\cdot\\mathcal{{S}}$ & {' & '.join(criterions)} \\\\")

if __name__ == '__main__':
    main()
