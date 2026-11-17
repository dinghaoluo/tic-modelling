# -*- coding: utf-8 -*-
'''
Created on Thu Nov  5 10:01:06 2026

check mean-trace scores and paired bootstrap trial multiplicities

@author: Dinghao Luo
'''

#%% imports
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.mean_trace_scores import mean_trace_scores, resample_weights

#%% tests
def test_mean_scores_use_same_trials_and_bin_threshold():
    signal = np.tile(np.array([1., 2., 4.]), (12, 1))
    signal[:3, 2] = np.nan
    prediction = signal + .2
    ll, r2, bins = mean_trace_scores(signal, prediction)
    observed_values = np.array([1., 2.])
    sse = np.sum(np.full_like(observed_values, .2) ** 2)
    np.testing.assert_allclose(ll, -len(observed_values) / 2 * np.log(sse / len(observed_values)))
    np.testing.assert_allclose(r2, 1 - sse / np.sum((observed_values - observed_values.mean()) ** 2))
    assert bins[0] == len(observed_values)


def test_weighted_bootstrap_matches_explicit_trial_resampling():
    rng = np.random.default_rng(19)
    signal = rng.normal(size=(16, 4))
    signal[:5, -1] = np.nan
    prediction = signal * .8 + .3
    samples = rng.integers(0, len(signal), (5, len(signal)))
    weights = resample_weights(samples, len(signal))
    actual = mean_trace_scores(signal, prediction, weights)
    for i, indices in enumerate(samples):
        expected = mean_trace_scores(signal[indices], prediction[indices])
        for actual_value, expected_value in zip(actual, expected):
            np.testing.assert_allclose(actual_value[i], expected_value[0])
