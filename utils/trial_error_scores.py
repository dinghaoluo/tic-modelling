'''Created on 5 November 2026
paired trial errors and mean-trace fits in both lick alignments
@author: Dinghao Luo
'''

#%% imports
import numpy as np

from utils.mean_trace_scores import mean_trace_scores

#%% scores
def trial_mse(signal, prediction):
    '''give each held-out trial equal weight, regardless of its observed duration'''
    observed = np.isfinite(signal)
    if prediction.shape != signal.shape or not np.isfinite(prediction[observed]).all():
        raise ValueError('prediction does not cover the observed bins')
    counts = observed.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError('trial has no observed bins')
    return np.where(observed, (signal - prediction) ** 2, 0.).sum(axis=1) / counts


def paired_gain(candidate_mse, reference_mse, resamples):
    '''positive means the candidate has smaller error; resample the paired differences'''
    difference = reference_mse - candidate_mse
    draws = difference[resamples].mean(axis=1)
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return dict(gain=float(difference.mean()), lower=float(lower), upper=float(upper))


def first_lick_align(signal, times, intervals):
    '''assign each observed bin centre to a 50 ms bin counted back from the first lick

    the original observations are retained without interpolation; each trial can
    contribute at most once to each bin in this alignment
    '''
    aligned = np.full(signal.shape, np.nan, dtype=float)
    trial, bins = np.where(np.isfinite(signal))
    back = np.floor((intervals[trial] - times[bins]) / .05 + 1e-10).astype(int)
    if np.any(back < 0) or np.any(back >= signal.shape[1]):
        raise ValueError('observed bin centre falls outside the aligned interval')
    if len(np.unique(trial * signal.shape[1] + back)) != len(trial):
        raise ValueError('more than one observation from a trial in an aligned bin')
    aligned[trial, back] = signal[trial, bins]
    return aligned


def alignment_scores(signal, prediction, times, intervals):
    ll, last_r2, last_bins = mean_trace_scores(signal, prediction)
    first_signal = first_lick_align(signal, times, intervals)
    first_prediction = first_lick_align(prediction, times, intervals)
    _, first_r2, first_bins = mean_trace_scores(first_signal, first_prediction)
    return dict(held_ll=float(ll[0]), held_r2=float(last_r2[0]),
                held_r2_last=float(last_r2[0]), held_r2_first=float(first_r2[0]),
                score_bins_last=int(last_bins[0]), score_bins_first=int(first_bins[0]))
