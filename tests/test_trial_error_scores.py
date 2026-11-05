'''Created on 5 November 2026
check negative centres, paired error signs and first-lick bin membership
@author: Dinghao Luo
'''

#%% imports
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.rf_models_v1_2 import BOUNDS, fit_field, predict
from utils.trial_error_scores import paired_gain, trial_mse, first_lick_align, alignment_scores

#%% tests
@pytest.mark.parametrize('name', ['gaussian', 'decay', 'rise'])
def test_negative_mu_recovery(name):
    '''the revised lower bound must admit a generating centre before zero'''
    interval = np.repeat([3.7, 4.1, 4.6], 70)
    t = np.concatenate([np.linspace(.024, end - .026, 70) for end in [3.7, 4.1, 4.6]])
    params = np.array([.2, 1.4, -.15, .7 if name == 'gaussian' else 1.7])
    fitted, success = fit_field(name, t, interval, predict(name, t, interval, params))
    assert BOUNDS[name][0][0] == -1
    assert success
    np.testing.assert_allclose(fitted, params, rtol=2e-4, atol=2e-5)


def test_paired_sign_and_equal_trial_weight():
    '''short and long trials have equal weight, and positive favours the candidate'''
    signal = np.array([[0., 0., np.nan], [0., 0., 0.]])
    candidate = np.array([[1., 1., np.nan], [2., 2., 2.]])
    reference = np.array([[2., 2., np.nan], [3., 3., 3.]])
    error = trial_mse(signal, candidate)
    np.testing.assert_array_equal(error, [1., 4.])
    resamples = np.array([[0, 0], [0, 1], [1, 1]])
    gain = paired_gain(error, trial_mse(signal, reference), resamples)
    assert gain['gain'] == 4
    assert gain['lower'] > 0
    reverse = paired_gain(trial_mse(signal, reference), error, resamples)
    np.testing.assert_allclose([reverse['lower'], reverse['upper']], [-gain['upper'], -gain['lower']])


def test_first_alignment_preserves_observations():
    signal = np.array([[1., 2., 3., np.nan], [4., 5., 6., np.nan]])
    times = np.array([.024, .074, .124, .174])
    aligned = first_lick_align(signal, times, np.array([.15, .17]))
    np.testing.assert_equal(aligned[:, :3], signal[:, :3][:, ::-1])
    assert np.isfinite(aligned).sum() == np.isfinite(signal).sum()


def test_both_scores_use_ten_trial_threshold():
    signal = np.tile([1., 2., 4., np.nan], (10, 1))
    scores = alignment_scores(signal, signal.copy(), np.array([.024, .074, .124, .174]), np.full(10, .15))
    assert scores['held_r2_last'] == scores['held_r2_first'] == 1
    assert scores['score_bins_last'] == scores['score_bins_first'] == 3
