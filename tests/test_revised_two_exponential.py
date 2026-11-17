# -*- coding: utf-8 -*-
'''
Created on Tue Nov 17 12:45:08 2026

check revised bump endpoints, clipped onsets and plain exponential mirroring

@author: Dinghao Luo
'''

#%% imports
import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from analysis.revised_two_exponential import (predict_model, model_shape, peak_and_onset,
                                         fit_model)

#%% tests
@pytest.mark.parametrize('anchoring', ['last', 'next'])
def test_revised_endpoints_and_empty_pieces(anchoring):
    params = np.array([.2, 1.4, -.4, 1.3, .7, 2.])
    T = 4.
    peak, onset = peak_and_onset(np.array(T), params[2:], anchoring)
    t = np.array([onset - .1, onset, peak, T])
    np.testing.assert_allclose(predict_model('revised_' + anchoring, t, T, params), [.2, .2, 1.6, .2])
    for t0 in [-1., 0., 3.]:
        for u in [0., 4., 7.]:
            params[2:4] = [t0, u]
            t = np.linspace(-1., T, 101)
            assert np.isfinite(predict_model('revised_' + anchoring, t, T, params)).all()
            np.testing.assert_allclose(predict_model('revised_' + anchoring, np.array([T]), T, params), [.2])
    params[2:4] = [3., 1.]
    peak, onset = peak_and_onset(np.array([1., 2., 4.]), params[2:], anchoring)
    assert (onset <= peak).all()
    if anchoring == 'last':
        np.testing.assert_allclose(onset, [1., 1., 1.])
    else:
        np.testing.assert_allclose(predict_model('revised_next', np.array([0., .5, 1.]), 1., params), .2)


def test_plain_is_the_written_formula_and_mirrors():
    params = np.array([.2, 1.4, .3, 2.])
    t = np.array([-1., .3, 1., 4.])
    expected = .2 + 1.4 * np.exp(-np.maximum(t - .3, 0.) / 2.)
    np.testing.assert_allclose(predict_model('plain', t, 4., params), expected)
    np.testing.assert_allclose(predict_model('plain', 4. - t, 4., params, True), expected)
    for mu in [-1., 3.]:
        for tau in [.1, 10.]:
            assert np.isfinite(predict_model('plain', t, 4., [.2, 1.4, mu, tau], True)).all()


@pytest.mark.parametrize('model, params, mirrored', [
    ('revised_last', [-.3, 1.3, .7, 2.], False),
    ('revised_next', [-.3, 1.3, .7, 2.], False),
    ('revised_last', [1.8, 1.3, .7, 2.], False),
    ('revised_next', [1.8, 1.3, .7, 2.], False),
    ('plain', [.3, 2.], False), ('plain', [.3, 2.], True)])
def test_shape_jacobian_matches_finite_differences(model, params, mirrored):
    '''the profiled refinement needs derivatives through trial-specific clipping'''
    t = np.linspace(-.98, 3.98, 87)
    T = np.linspace(1.1, 4.3, len(t))
    _, jac = model_shape(model, t, T, params, mirrored, True)
    numeric = approx_derivative(lambda p: model_shape(model, t, T, p, mirrored), params)
    np.testing.assert_allclose(jac, numeric, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('model, params', [
    ('revised_last', [.2, 1.4, -.4, 1.3, .7, 2.]),
    ('revised_next', [.2, 1.4, -.4, 1.3, .7, 2.]),
    ('plain', [.2, 1.4, .3, 2.])])
def test_profiled_fit_recovers_interior_curve(model, params):
    t = np.tile(np.linspace(-.97, 5.9, 100), 3)
    T = np.repeat([3.8, 4.7, 6.], 100)
    t, T = t[t < T], T[t < T]
    y = predict_model(model, t, T, params)
    fitted, ok, _ = fit_model(model, t, T, y, np.ones(len(t)))
    assert ok and fitted[1] >= 0
    np.testing.assert_allclose(predict_model(model, t, T, fitted), y, atol=1e-5)
