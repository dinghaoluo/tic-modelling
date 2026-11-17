'''
2026-11-09
tests for bound categories, directional peaks and source session order
@author: Dinghao Luo
'''

#%% imports
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.paper_parameter_comparison import fitted_values
from analysis.read_only_diagnostics import load_session_order
from analysis.summary_figure import parameter_name
from analysis.wider_parameter_summary import bound_percentages, final_bin_proportion

#%% tests
def test_bound_percentages_split_lower_and_upper():
    parameters = pd.DataFrame({
        'group': ['tic_down', 'tic_down', 'tic_down'],
        'decay_tau': [.1, 10., 2.],
        'decay_tau_at_bound': [True, True, False],
    })
    lower, upper = bound_percentages(parameters, 'tic_down', 'decay_tau')
    assert lower == 100. / 3
    assert upper == 100. / 3


def test_tic_peak_parameter_names_are_directional():
    assert parameter_name('tic_down', 'peak') == 'decay_mu'
    assert parameter_name('tic_up', 'peak') == 'rise_mu'
    assert parameter_name('time_cell', 'peak') == 'gaussian_mu'


def test_session_order_parser_ignores_comments_and_duplicates(tmp_path):
    source = tmp_path / 'PathFor4sData.m'
    source.write_text("""
SessionsPath4s = [...
 {'A132-20221124-02'}
 % {'A132-20221124-04'}
 {'A132-20221124-04'}
]
""")
    assert load_session_order(source) == ['A132-20221124-02', 'A132-20221124-04']



def test_final_bin_includes_upper_bound_values():
    assert np.isclose(final_bin_proportion([9., 10., 10.], 'Time constant \u03c4', [False, True, True]), 200. / 3)
    assert np.isclose(final_bin_proportion([3.5, 4., 4.], 'Gaussian \u03c3', [False, True, True]), 200. / 3)


def test_ramp_up_peak_is_signed_for_cao_convention():
    parameters = pd.DataFrame({'group': ['tic_up'], 'rise_mu': [2.],
                               'rise_mu_at_bound': [False]})
    values, source = fitted_values(parameters, 'tic_up', 'peak', True)
    assert values.tolist() == [-2.]
    assert source == '-rise_mu'
