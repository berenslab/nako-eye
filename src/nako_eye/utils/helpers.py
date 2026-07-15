import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats
from scipy.stats import ks_2samp
from sklearn.metrics import cohen_kappa_score
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


def muks_1d(
    scores_p,
    scores_q,
    sample_size,
    num_repetitions=100,
    alpha=0.05,
    replace=False,
    seed=42,
):
    """MUKS-style shift test for a 1D score (Koch et al. 2022, MIDL).

    Repeatedly runs a two-sample KS test on resampled subsets to compare
    P vs Q (test power) and P vs P (empirical Type I error, should be
    close to `alpha` if well-calibrated).

    Parameters
    ----------
    scores_p, scores_q : np.ndarray
        1D scores from domains P and Q.
    sample_size : int
        Examples drawn per domain per repetition. If replace=False, must
        be <= n_p // 2 (P is split into two disjoint draws) and <= n_q.
    num_repetitions : int, default=100
    alpha : float, default=0.05
    replace : bool, default=False
        Whether to sample with replacement.
    seed : int, default=42

    Returns
    -------
    reject_rate, reject_rate_h0 : float
        Power (P vs Q) and empirical Type I error (P vs P).
    """
    rng = np.random.default_rng(seed)
    n_p, n_q = len(scores_p), len(scores_q)

    if not replace:
        assert 2 * sample_size <= n_p, (
            'sample_size must be <= n_p // 2 when replace=False (need two disjoint draws from P)'
        )
        assert sample_size <= n_q, (
            'sample_size exceeds pool size for Q (set replace=True to allow)'
        )

    count_rejects = 0  # power: P vs Q
    count_rejects_h0 = 0  # type I error: P vs P

    for _ in range(num_repetitions):
        if replace:
            idx_p = rng.choice(n_p, size=sample_size, replace=True)
            idx_p2 = rng.choice(n_p, size=sample_size, replace=True)
        else:
            # Disjoint draws, so the P-vs-P comparison isn't biased by shared samples
            perm_p = rng.permutation(n_p)
            idx_p, idx_p2 = perm_p[:sample_size], perm_p[sample_size : 2 * sample_size]
        idx_q = rng.choice(n_q, size=sample_size, replace=replace)

        x = scores_p[idx_p]
        x2 = scores_p[idx_p2]
        y = scores_q[idx_q]

        _, pval = ks_2samp(x, y)
        _, pval0 = ks_2samp(x, x2)

        count_rejects += pval < alpha
        count_rejects_h0 += pval0 < alpha

    reject_rate = count_rejects / num_repetitions
    reject_rate_h0 = count_rejects_h0 / num_repetitions

    return reject_rate, reject_rate_h0


def kappa_test(y1, y2, weights=None, labels=None, n_perm=10000, seed=42):
    """Cohen's kappa (optionally weighted) with a permutation-test p-value.

    Parameters
    ----------
    y1, y2 : array-like
        Ratings from two raters (categorical, e.g. binary or ordinal).
    weights : {None, 'linear', 'quadratic'}, default=None
        Passed to `sklearn.metrics.cohen_kappa_score`.
    labels : array-like, optional
        Category order, passed to `sklearn.metrics.cohen_kappa_score`. Required
        for weighted kappa on non-integer categories, since `cohen_kappa_score`
        otherwise orders categories alphabetically rather than by ordinal rank.
    n_perm : int, default=10000
        Number of label permutations used to build the null distribution.
    seed : int, default=42

    Returns
    -------
    kappa, pvalue : float
    """
    rng = np.random.default_rng(seed)
    y1, y2 = np.asarray(y1), np.asarray(y2)

    if np.unique(y1).size < 2 or np.unique(y2).size < 2:
        raise ValueError('kappa is undefined when a rater has only one category')

    kappa = cohen_kappa_score(y1, y2, labels=labels, weights=weights)
    perm_kappas = np.array(
        [
            cohen_kappa_score(y1, rng.permutation(y2), labels=labels, weights=weights)
            for _ in range(n_perm)
        ]
    )
    pvalue = (np.sum(np.abs(perm_kappas) >= np.abs(kappa)) + 1) / (n_perm + 1)
    return kappa, pvalue


def majority_vote(ratings, tie_break):
    """Most frequent rating; ties broken by whichever tied value is closest to `tie_break`."""
    counts = ratings.value_counts()
    modes = counts[counts == counts.max()].index
    return (
        int(modes[0])
        if len(modes) == 1
        else int(min(modes, key=lambda x: abs(x - tie_break)))
    )


if __name__ == '__main__':
    print('utils.py')
