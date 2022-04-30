import re
import sys
from collections import defaultdict

def display_arch(arch):
    return {
        'resnet18': "ResNet-18",
        'vgg8': "VGG8",
        'sfcn': "SFCN",
        'glt': "GLT",
        '3dt': "3D Transformer"
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

def write_header(arch):
    print(f"        \multirow{{13}}{{*}}{{{display_arch(arch)}}} \\\\")

def write_results(strat, results):
    print(f"        & {display_strat(strat)} & {results[0]} & {results[1]} & {results[2]} & {results[3]} \\\\ \\cline{{2-6}}")

def main():
    input_file = sys.argv[1]
    with open(input_file, 'r') as f:
        to_parse = f.read()
    
    sections = to_parse.split('\n======\n')
    d = defaultdict(dict)
    for section in sections:
        sec_lines = section.split('\n')
        if len(sec_lines) != 4:
            continue

        overall_mae = float(sec_lines[0].split(' ')[-1])
        rho = float(sec_lines[1].split(' ')[-1])
        entropy = float(sec_lines[2].split(' ')[-1])
        criterion = float(sec_lines[3].split(' ')[-1])

        m = re.match(r'.* for (.*) \((.*)\).*', sec_lines[0])
        strat, arch = m[1], m[2]

        d[arch][strat] = (overall_mae, rho, entropy, criterion)

    for arch in ('resnet18', 'vgg8', 'glt', 'sfcn', '3dt'):
        write_header(arch)
        for strat in ('baseline', 'under', 'scale-down', 'over', 'scale-up', 'inv', 'lds+inv', 'fds+inv', 'lds+fds+inv', 'sqrt_inv', 'lds+sqrt_inv', 'fds+sqrt_inv', 'lds+fds+sqrt_inv'):
            if strat not in d[arch]:
                continue
            results = d[arch][strat]
            write_results(strat, results)

if __name__ == '__main__':
    main()
