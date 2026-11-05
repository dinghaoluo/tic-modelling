'''Created on 5 November 2026
check trial splits, drop-top pairing and binned trace reliability
@author: Dinghao Luo
'''

#%% imports
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.label_stability import lick_ratio, trial_masks, split_correlation

#%% tests
def test_trial_masks_and_stable_top_removal():
    '''odd rows are one-based and top trials are removed using last-lick activity'''
    arr = np.zeros((11, 5000))
    arr[3, 1250:1750] = 10
    arr[7, 1250:1750] = 10
    masks, removed = trial_masks(arr)
    np.testing.assert_array_equal(np.flatnonzero(masks['odd']), [0, 2, 4, 6, 8, 10])
    np.testing.assert_array_equal(np.flatnonzero(masks['even']), [1, 3, 5, 7, 9])
    np.testing.assert_array_equal(removed, [3, 7])
    assert not masks['drop_top'][removed].any()
    assert masks['drop_top'].sum() == len(arr) - len(removed)


def test_full_profile_normalisation_and_ratio():
    profile = np.full(5000, 2.)
    profile[0] = 0
    profile[-1] = 10
    profile[1500:1750] = 4
    assert lick_ratio(profile) == 2


def test_binned_split_correlation():
    '''reliability ignores the flanking samples and averages within each 50 ms bin'''
    odd = np.zeros(5000)
    even = np.zeros(5000)
    odd[1500:3500] = np.repeat(np.arange(80), 25)
    even[1500:3500] = 3 * odd[1500:3500] + 2
    even[:1500] = -1000
    np.testing.assert_allclose(split_correlation(odd, even), 1)
    assert np.isnan(split_correlation(np.zeros(5000), even))
