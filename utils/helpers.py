import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats
from sklearn.utils import resample

SEED = 42


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_image(filename, return_np=True):
    """Load image file as a PIL image or a numpy array."""
    image = Image.open(filename)
    image.load()
    data = np.asarray(image, dtype='int32')
    return data if return_np else image


def sort_categorical_df(df, column_name, ascending=False):
    """Sort dataframe based on the counts for each category of a column that holds a categorical variable."""
    df['sort_col'] = df[column_name].map(df[column_name].value_counts())
    df = df.sort_values(by='sort_col', ascending=ascending).drop('sort_col', axis=1)
    return df


def add_row_csv(source_file, dest_file, index_col='ID', row_idx=0, fill=np.nan):
    df = pd.read_csv(source_file, index_col=index_col)
    df.loc[row_idx] = fill
    df = df.sort_index()
    df.to_csv(dest_file, mode='w', index=True, header=True)


def bootstrap_test(preds, targets, metrics, n_trials=1000, seed=42):
    """Bootstrap test set and return the average performance and 95% confidence interval"""
    random_state = np.random.RandomState(seed=seed)
    metrics_ = np.zeros((n_trials, len(metrics)))
    for n in range(n_trials):
        preds_sample, targets_sample = resample(
            preds, targets, replace=True, random_state=random_state
        )

        for m, metric in enumerate(metrics):
            metrics_[n, m] = metric(targets_sample, preds_sample)

    for m, metric in enumerate(metrics):
        perc = np.percentile(metrics_[:, m], [2.5, 97.5])
        print(
            f'{metric.__name__}: {metrics_[:, m].mean():.3f} ({perc[0]:.3f} - {perc[1]:.3f})'
        )


def remove_outliers_iqr(df, column, percentiles=[25, 75]):
    """Remove outliers outside the IQR (Q3 -Q1)."""
    perc = np.nanpercentile(df[column], percentiles)
    iqr = perc[1] - perc[0]
    upper_limit = perc[1] + 1.5 * iqr
    lower_limit = perc[0] - 1.5 * iqr

    df_subset = df[(df[column] > lower_limit) & (df[column] < upper_limit)]
    return df_subset


def remove_outliers_z(df, column, threshold=3):
    z = np.abs(stats.zscore(df[column]))
    df_subset = df[z < threshold]
    return df_subset


if __name__ == '__main__':
    print('utils.py')
