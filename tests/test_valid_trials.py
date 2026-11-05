'''Created on 4 November 2026
check MATLAB trial numbering and strict validity thresholds
@author: Dinghao Luo
'''

#%% imports
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.compute_valid_trials import valid_trial_rows

#%% tests
def test_adjacent_trials_and_uncomputed_edges():
    '''trial one and the final trial stay zero even if their licks would pass'''
    start = np.arange(7) * 3000.
    end = start + 2000.
    licks = [np.array([s + 600., s + 1200.]) for s in start]
    licks[3] = np.array([])
    expected_passes = np.array([False, True, True, False, True, True, False])
    expected = np.flatnonzero(expected_passes[:-1] & expected_passes[1:]) + 1
    valid, last, first = valid_trial_rows(start, end, licks)
    np.testing.assert_array_equal(valid, expected)
    assert last[0] == first[0] == last[-1] == first[-1] == 0
    assert first[3] < 0


def test_thresholds_are_strict():
    start = np.arange(5) * 3000.
    end = start + 2000.
    licks = [np.array([s + 600., s + 1200.]) for s in start]
    licks[1] = np.array([start[1] + 500., start[1] + 1200.])
    licks[2] = np.array([start[2] + 600., end[2] - 450.])
    expected_passes = np.array([False, False, False, True, False])
    expected = np.flatnonzero(expected_passes[:-1] & expected_passes[1:]) + 1
    valid, _, _ = valid_trial_rows(start, end, licks)
    np.testing.assert_array_equal(valid, expected)

def test_available_source_lists_match_computed_rows():
    '''source expectations stay in the MATLAB tables rather than in this test'''
    from rec_list import SESSIONS_4S, session_path
    from utils.matlab_inputs import load_numeric

    root = Path(__file__).resolve().parents[1]
    compared = []
    for session in SESSIONS_4S:
        source = session_path(session) / 'ValidTr_FirstLastLick2.mat'
        if not source.is_file():
            continue
        expected = load_numeric(source)['ValidTrAll'].ravel()
        actual = np.load(root / 'data' / 'valid_trials' / f'{session}.npy')
        np.testing.assert_array_equal(actual, expected)
        compared.append(session)
    assert compared
