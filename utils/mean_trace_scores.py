'''Created on 5 November 2026
score out-of-fold trial predictions after averaging the same observed bins
@author: Dinghao Luo
'''

#%% imports
import numpy as np

#%% scores
def mean_trace_scores(signal, prediction, weights=None):
    '''recompute eligible bins and both trial means for every bootstrap resample'''
    observed = np.isfinite(signal)
    if prediction.shape != signal.shape or np.any(~np.isfinite(prediction[observed])):
        raise ValueError('prediction does not cover every observed trial bin')
    if weights is None:
        weights = np.ones((1, len(signal)))
    counts = weights @ observed.astype(float)
    eligible = counts >= 10
    if np.any(eligible.sum(axis=1) == 0):
        raise ValueError('no bins have at least ten resampled trials')
    y_sum = weights @ np.where(observed, signal, 0.)
    p_sum = weights @ np.where(observed, prediction, 0.)
    y_mean = np.divide(y_sum, counts, out=np.zeros_like(y_sum), where=counts > 0)
    p_mean = np.divide(p_sum, counts, out=np.zeros_like(p_sum), where=counts > 0)
    n_bins = eligible.sum(axis=1)
    sse = np.sum(np.where(eligible, (y_mean - p_mean) ** 2, 0.), axis=1)
    centre = np.sum(np.where(eligible, y_mean, 0.), axis=1) / n_bins
    sst = np.sum(np.where(eligible, (y_mean - centre[:, None]) ** 2, 0.), axis=1)
    r2 = 1 - np.divide(sse, sst, out=np.full_like(sse, np.nan), where=sst > 0)
    with np.errstate(divide='ignore'):
        loglik = -n_bins / 2 * np.log(sse / n_bins)
    return loglik, r2, n_bins


def resample_weights(resamples, n_trials):
    weights = np.zeros((len(resamples), n_trials))
    for i, indices in enumerate(resamples):
        weights[i] = np.bincount(indices, minlength=n_trials)
    return weights
