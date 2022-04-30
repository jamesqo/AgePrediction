import os

import pandas as pd

from plot_results import load_results

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SHENG_DIR = os.path.join(ROOT_DIR, "results_for_sheng")

def main():
    os.makedirs(SHENG_DIR, exist_ok=True)
    results = load_results()

    archs = [
        'resnet18',
        'vgg8',
        'glt',
        'sfcn',
        '3dt'
    ]

    strats = [
        'baseline',
        'under',
        'scale-down',
        'over',
        'scale-up',
        'inv',
        'lds+inv',
        'lds+fds+inv',
        'sqrt_inv',
        'lds+sqrt_inv',
        'lds+fds+sqrt_inv'
    ]

    for arch in archs:
        for strat in strats:
            df = results[f"{strat} ({arch})"]['val_preds']
            pathkey = 'img_path' if 'img_path' in df else 'path'
            out_df = pd.DataFrame({
                'Filename': df[pathkey],
                'GroundTruth': df['age'],
                'Prediction': df['age_pred'],
                'Sex': df['sex']
            })
            outpath = os.path.join(ROOT_DIR, "results_for_sheng", f"Model__{arch}__{strat.replace('+', '__')}.csv")
            out_df.to_csv(outpath, index=False)

if __name__ == '__main__':
    main()
