# -*- coding: utf-8 -*-
'''
Created on Thu Nov  5 19:15:36 2026

check that peak sorting never evaluates the trials used for classification

@author: Dinghao Luo
'''

#%% imports
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.peak_position_check import trial_halves, half_bootstrap, peak_class, holm

#%% tests
def test_sorting_and_testing_trials_never_overlap():
    rows = np.array([0, 2, 3, 6, 7, 9])
    odd, even = trial_halves(rows)
    np.testing.assert_array_equal(rows[odd], [0, 2, 6])
    for sorting, testing in [(odd, even), (even, odd)]:
        assert not set(rows[sorting]) & set(rows[testing])
        values = np.where(testing, 2., 10000.)
        resamples = np.array([np.arange(len(rows)), np.arange(len(rows))[::-1]])
        mean, draws = half_bootstrap(values, testing, resamples)
        assert mean == 2.
        np.testing.assert_array_equal(draws, [2., 2.])


def test_peak_clipping_and_holm():
    assert peak_class(-.5, 1., 4.) == (0., 'start')
    assert peak_class(2., 1., 4.) == (2., 'inside')
    assert peak_class(7., 1., 4.) == (4., 'end')
    np.testing.assert_allclose(holm([.02, .01, .5]), [.04, .03, .5])
