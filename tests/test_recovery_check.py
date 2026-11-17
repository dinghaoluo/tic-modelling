# -*- coding: utf-8 -*-
'''
Created on Thu Nov  5 18:24:50 2026

check that trial sign flips preserve the observed residual time courses

@author: Dinghao Luo
'''

#%% imports
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.recovery_check import sign_flip_signal

#%% test
def test_trial_sign_flip_preserves_residual_pattern():
    truth = np.array([[2., 3., np.nan], [4., 5., 6.]])
    prediction = np.array([[1., 1., np.nan], [2., 2., 2.]])
    real = np.array([[2., 4., np.nan], [3., 5., 7.]])
    synthetic = sign_flip_signal(truth, real, prediction, np.array([1, -1]))
    np.testing.assert_equal(synthetic, [[3., 6., np.nan], [3., 2., 1.]])
    np.testing.assert_array_equal(np.isfinite(synthetic), np.isfinite(real))
    np.testing.assert_allclose(np.nansum((synthetic - truth) ** 2, axis=1),
                               np.nansum((real - prediction) ** 2, axis=1))
    np.testing.assert_equal(sign_flip_signal(truth, real, prediction, np.ones(2)),
                            truth + real - prediction)
