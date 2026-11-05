'''Created on 5 November 2026
check the sign and equal-trial weighting of saved-prediction likelihood contrasts
@author: Dinghao Luo
'''

#%% imports
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.compare_models_trial import trial_loglikelihood, trial_contrast
from utils.trial_error_scores import trial_mse

#%% test
def test_per_trial_likelihood_contrast():
    '''the common variance cancels constants without weighting longer trials more'''
    y = np.array([[0., 0., np.nan], [0., 0., 0.]])
    a = np.array([[1., 1., np.nan], [2., 2., 2.]])
    b = np.array([[2., 2., np.nan], [3., 3., 3.]])
    mse_a, mse_b = trial_mse(y, a), trial_mse(y, b)
    variance = 2.
    difference = trial_contrast(mse_a, mse_b, variance)
    np.testing.assert_allclose(difference, [.75, 1.25])
    np.testing.assert_allclose(difference, trial_loglikelihood(mse_a, variance) - trial_loglikelihood(mse_b, variance))
    assert difference.mean() == 1
    np.testing.assert_allclose(trial_contrast(mse_b, mse_a, variance), -difference)
