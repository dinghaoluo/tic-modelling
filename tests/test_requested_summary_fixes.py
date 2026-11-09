'''
2026-11-09
tests for bound categories, directional peaks and source session order
@author: Dinghao Luo
'''

#%% imports
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.read_only_diagnostics import load_session_order
from analysis.summary_figure import parameter_name
from analysis.wider_parameter_summary import bound_percentages

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
