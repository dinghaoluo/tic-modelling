'''Created on 5 November 2026
check that likelihood contrasts choose the better exponential per cell
@author: Dinghao Luo
'''

#%% imports
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.compare_models import likelihood_contrasts

#%% test
def test_likelihood_contrasts():
    frame = pd.DataFrame({'gaussian_held_ll': [8., 4., -2.],
                          'decay_held_ll': [3., 7., -4.],
                          'rise_held_ll': [6., 1., -2.]}, index=[2, 4, 7])
    result = likelihood_contrasts(frame)
    np.testing.assert_array_equal(result.index, frame.index)
    np.testing.assert_allclose(result.gaussian_minus_decay, [5., -3., 2.])
    np.testing.assert_allclose(result.gaussian_minus_rise, [2., 3., 0.])
    np.testing.assert_allclose(result.gaussian_minus_better_exponential, [2., -3., 0.])
