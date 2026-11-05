'''Created on 4 November 2026
noiseless parameter recovery for the specified receptive-field shapes
@author: Dinghao Luo
'''

#%% imports
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.rf_models import fit_field, predict

#%% tests
@pytest.mark.parametrize('name, params', [
    ('flat', np.array([.3])),
    ('gaussian', np.array([.2, 1.4, 2.3, .7])),
    ('decay', np.array([.2, 1.4, .6, 1.7])),
    ('rise', np.array([.2, 1.4, .6, 1.7])),
    ])
def test_noiseless_recovery(name, params):
    '''trial-specific rise coordinates must recover the same generating parameters'''
    interval = np.repeat(np.array([3.7, 4.1, 4.6]), 70)
    t = np.concatenate([np.linspace(.025, end - .025, 70) for end in [3.7, 4.1, 4.6]])
    y = predict(name, t, interval, params)
    fitted, success = fit_field(name, t, interval, y)
    assert success
    np.testing.assert_allclose(fitted, params, rtol=1e-4, atol=1e-5)
