'''Created on 5 November 2026
check held-out row selection and epoch alignment without model fitting
@author: Dinghao Luo
'''

#%% imports
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.epoch_check import selected_rows, epoch_means, held_out_group

#%% tests
def test_row_parity_survives_interval_filter():
    '''odd/even parity belongs to the original rows, not the filtered list'''
    first = np.array([0., 2., 3., 4., 7., 6.998, 3.5])
    odd, even = selected_rows(first, len(first) - 1)
    np.testing.assert_array_equal(odd, [2, 4])
    np.testing.assert_array_equal(even, [1, 5])


def test_epochs_use_their_own_alignment_and_equal_trial_weights():
    '''different mid lengths should not give the longer trial extra weight'''
    last = np.zeros((2, 5000))
    first = np.zeros((2, 5000))
    last[:, 1250:1500] = 2
    last[:, 1500:2000] = 3
    last[0, 2000:2500] = 4
    last[1, 2000:4000] = 8
    first[:, 1000:1500] = 10
    first[:, 1500:1750] = 12
    values = epoch_means(last, first, np.array([3., 6.]))
    assert values == dict(pre=2., early=3., mid=6., late=10., post=12.)


def test_odd_selected_groups_keep_source_cluster():
    assert held_out_group(True, True, 'down') == 'odd_tic_down'
    assert held_out_group(True, True, 'up') == 'odd_tic_up'
    assert held_out_group(True, False, np.nan) == 'odd_tic_new'
    assert held_out_group(False, True, 'down') == 'not_tic'
