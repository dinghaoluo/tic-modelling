'''Created on 15 November 2026
check variance explained against the observed mean
@author: Dinghao Luo
'''

#%% imports
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.fit_quality import r_squared


#%% test
def test_r_squared_uses_observed_variance_and_shared_bins():
    '''an offset can give negative R²; missing and constant traces remain explicit'''
    observed = np.array([0., 1., 2.])
    assert r_squared(observed, observed) == 1.
    assert r_squared(observed, np.ones(3)) == 0.
    assert r_squared(observed, observed + 1.) == -.5
    assert r_squared([0., 1., np.nan, 50.], [0., 1., 100., np.nan]) == 1.
    assert np.isnan(r_squared([1., 1.], [1., 1.]))
    assert np.isnan(r_squared([np.nan, 1.], [0., 1.]))
    with pytest.raises(ValueError, match='same shape'):
        r_squared([0., 1.], [0.])
