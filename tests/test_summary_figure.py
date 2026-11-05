'''Created on 5 November 2026
check summary categories, p-value labels and parameter-specific exclusions
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
    percentages, p, n_sessions = category_summary(cells, 'poly4', 'poly1')
    np.testing.assert_allclose(percentages, [50., 25., 25.])
    assert n_sessions == 2 and 0 <= p <= 1
    assert p_label(.0001) == 'p < 0.001'
    assert p_label(.2) == 'p = 0.20 (n.s.)'


def test_parameter_exclusion_is_not_shape_exclusion():
    cells = pd.DataFrame({'group': ['tic_down', 'tic_down'], 'gaussian_mu': [1., 2.],
                          'gaussian_mu_at_bound': [False, True],
                          'gaussian_sigma_at_bound': [True, False]})
    kept, parameter, percentage = parameter_values(cells, 'tic_down', 'peak')
    assert parameter == 'gaussian_mu'
    assert percentage == 50
    np.testing.assert_array_equal(kept.gaussian_mu, [1.])
