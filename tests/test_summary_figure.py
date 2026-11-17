# -*- coding: utf-8 -*-
'''
Created on Thu Nov  5 18:05:57 2026

check summary categories, p-value labels and parameter-specific bound handling

@author: Dinghao Luo
'''

#%% imports
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.summary_figure import category_summary, p_label, parameter_values

#%% tests
def test_categories_sum_to_one_hundred():
    cells = pd.DataFrame({'session': ['a', 'a', 'b', 'b'],
                          'mean_contrast': [1., 2., -1., 0.],
                          'decision': ['poly4', 'poly4', 'poly1', 'undecided']})
    percentages, p_value, n_sessions = category_summary(cells, 'poly4', 'poly1')
    np.testing.assert_allclose(percentages, [50., 25., 25.])
    assert n_sessions == 2 and 0 <= p_value <= 1
    assert p_label(.0001) == 'p < 0.001'
    assert p_label(.2) == 'p = 0.20 (n.s.)'


def test_parameter_exclusion_is_not_shape_exclusion():
    cells = pd.DataFrame({'group': ['tic_down', 'tic_down'], 'decay_mu': [1., 2.],
                          'decay_mu_at_bound': [False, True],
                          'gaussian_sigma_at_bound': [True, False]})
    kept, parameter, percentage = parameter_values(cells, 'tic_down', 'peak')
    assert parameter == 'decay_mu'
    assert percentage == 50
    np.testing.assert_array_equal(kept.decay_mu, [1.])


def test_bound_inclusion_keeps_every_cell():
    cells = pd.DataFrame({'group': ['tic_down', 'tic_down'], 'gaussian_sigma': [.5, 4.],
                          'gaussian_sigma_at_bound': [False, True]})
    retained, parameter, percentage = parameter_values(cells, 'tic_down', 'width', include_bound=True)
    assert parameter == 'gaussian_sigma' and percentage == 50
    np.testing.assert_array_equal(retained[parameter], [.5, 4.])
    excluded, _, _ = parameter_values(cells, 'tic_down', 'width')
    np.testing.assert_array_equal(excluded[parameter], [.5])
