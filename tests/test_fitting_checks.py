# -*- coding: utf-8 -*-
'''
Created on Mon Nov 16 13:12:22 2026

check bump endpoints, exact compression and equal time-bin weights

@author: Dinghao Luo
'''

#%% imports
import numpy as np
import pytest
from analysis.two_exponential import predict_bump, compress_pairs, fit_bump
from utils.rf_models import fit_field, predict

#%% tests
@pytest.mark.parametrize('anchor', ['last', 'next'])
def test_bump_endpoints_and_peak(anchor):
    params = np.array([.2, 1.4, 1.3, .7, 2.])
    T = 4.
    peak = params[2] if anchor == 'last' else T - params[2]
    np.testing.assert_allclose(predict_bump(np.array([0., peak, T]), T, params, anchor), [.2, 1.6, .2])
    for u in [0., 4., 7.]:
        params[2] = u
        fitted = predict_bump(np.linspace(-1, T, 101), T, params, anchor)
        assert np.isfinite(fitted).all()
        np.testing.assert_allclose(fitted[[0, 20, -1]], .2)


def test_pair_compression_preserves_weighted_error():
    t = np.array([.2, .2, .2, 1., 1.])
    T = np.array([3., 3., 4., 3., 3.])
    y = np.array([1., 2., 3., 2., 4.])
    x, interval, mean, count = compress_pairs(t, T, y)
    offsets = []
    for p in [np.array([.2, 1., 1.3, .7, 2.]), np.array([.3, 2., .5, 1., 4.])]:
        raw = np.sum((y - predict_bump(t, T, p, 'last')) ** 2)
        compressed = np.sum(count * (mean - predict_bump(x, interval, p, 'last')) ** 2)
        offsets.append(raw - compressed)
    np.testing.assert_allclose(offsets[0], offsets[1])


def test_equal_time_weights_ignore_replication():
    t = np.linspace(-.9, 6.9, 61)
    y = predict('gaussian', t, np.zeros_like(t), np.array([.2, 1.2, 2., .8])) + .03 * np.sin(2. * t)
    repeats = np.arange(len(t)) % 7 + 1
    expanded_t = np.repeat(t, repeats)
    params, ok = fit_field('gaussian', expanded_t, np.zeros_like(expanded_t), np.repeat(y, repeats), equal_time_weight=True)
    reference, reference_ok = fit_field('gaussian', t, np.zeros_like(t), y)
    assert ok and reference_ok
    np.testing.assert_allclose(params, reference, atol=1e-7)

@pytest.mark.parametrize('anchor', ['last', 'next'])
def test_noiseless_bump_recovery(anchor):
    '''the profiled fit recovers an interior bump without dropping trial bins'''
    t = np.tile(np.linspace(-.8, 5.9, 120), 3)
    interval = np.repeat([3.8, 4.7, 6.], 120)
    keep = t < interval
    t, interval = t[keep], interval[keep]
    params = np.array([.2, 1.4, 1.3, .7, 2.])
    values = predict_bump(t, interval, params, anchor)
    x, T, y, weights = compress_pairs(t, interval, values)
    fitted, ok, _ = fit_bump(x, T, y, weights, anchor)
    assert ok
    np.testing.assert_allclose(fitted, params, atol=1e-5, rtol=1e-5)


def test_thresholds_use_average_normalisation_and_paired_ranking():
    '''single trials can have negative normalised pre-event values, unlike the average'''
    from analysis.trial_threshold import profile_pass, trial_pass, removed_trials
    array = np.full((2, 5000), 2.)
    array[0, 1250:1500] = 0.
    array[0, 1500:1750] = 1.
    array[1, 1250:1500] = 4.
    array[1, 1500:1750] = 5.
    profile = array.mean(0)
    minimum, span = profile.min(), np.ptp(profile)
    normalised = (array - minimum) / span
    pre = normalised[:, 1250:1500].mean(1)
    post = normalised[:, 1500:1750].mean(1)
    np.testing.assert_array_equal(trial_pass(array, True), post - 1.5 * pre > 0)
    np.testing.assert_array_equal(trial_pass(array, False), post - (2/3) * pre < 0)
    assert pre[0] < 0
    np.testing.assert_array_equal(removed_trials(array, .1), [1])
    passed, defined = profile_pass(np.ones(5000), True)
    assert not passed[0] and not defined[0]


def test_ten_trial_support_is_on_the_models_coordinate():
    from analysis.equal_weight_check import retained_coordinates
    t = np.repeat([0., 1., 2.], [9, 10, 11])
    interval = np.full(len(t), 4.)
    x, T, y = retained_coordinates('gaussian', t, interval, np.ones(len(t)))
    np.testing.assert_array_equal(x, [1., 2.])
    x, T, y = retained_coordinates('rise', t, interval, np.ones(len(t)), np.arange(len(t)))
    np.testing.assert_allclose(T, [2.025, 3.025])


def test_rise_bins_pool_jitter_but_count_distinct_trials():
    from analysis.equal_weight_check import retained_coordinates
    trial = np.r_[np.arange(10), np.arange(9), np.zeros(20, dtype=int)]
    coordinate = np.r_[np.linspace(.101, .149, 10), np.linspace(.201, .249, 9), np.full(20, .31)]
    values = np.r_[np.arange(10), np.ones(9), np.ones(20)]
    t, interval, y = retained_coordinates('rise', np.zeros(len(trial)), coordinate, values, trial)
    np.testing.assert_allclose(t, [0.])
    np.testing.assert_allclose(interval, [.125])
    np.testing.assert_allclose(y, [4.5])


def test_extra_curve_uses_the_same_averaging_operator():
    from types import SimpleNamespace
    from analysis.fit_quality import average_traces
    cell = SimpleNamespace(group='tic_up', session='synthetic', neu_id=1,
                           gaussian_a0=.2, gaussian_a=1., gaussian_mu=1., gaussian_sigma=.5,
                           rise_a0=.2, rise_a=1., rise_mu=.5, rise_tau=1.)
    t = np.arange(-.975, 4., .05)
    T = np.linspace(2., 3.5, 15)
    signal = np.broadcast_to(np.sin(t), (len(T), len(t))).copy()
    signal[t[None, :] > T[:, None]] = np.nan
    extra = np.where(np.isfinite(signal), 2. * signal + 3., 0.)
    for alignment in ['last_lick', 'next_lick']:
        x, observed, _, _, fitted = average_traces(signal, t, T, cell, alignment, additional=[extra])
        np.testing.assert_allclose(fitted, 2. * observed + 3., atol=1e-12)


def test_trial_contrast_keeps_the_candidate_sign():
    from analysis.two_exponential import contrast_row
    resamples = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    row = contrast_row(np.array([1., 1.]), np.array([3., 3.]), 2., resamples)
    assert row['mean_contrast'] == .5
    assert row['lower'] == row['upper'] == .5
    assert row['decision'] == 'favour_a'


def test_width_override_keeps_the_default_bound():
    '''the wider grid and bounds must not replace the default cached Gaussian fit'''
    t = np.linspace(-1., 7., 160)
    T = np.zeros_like(t)
    y = predict('gaussian', t, T, np.array([.2, 1.4, 2.3, 6.]))
    ordinary, ok = fit_field('gaussian', t, T, y)
    wide, wide_ok = fit_field('gaussian', t, T, y, bounds=(np.array([-1., .05]), np.array([7., 10.])))
    repeated, repeated_ok = fit_field('gaussian', t, T, y)
    assert ok and wide_ok and repeated_ok
    assert ordinary[3] <= 4.
    np.testing.assert_allclose(wide, [.2, 1.4, 2.3, 6.], atol=1e-4)
    np.testing.assert_array_equal(ordinary, repeated)
