'''Created on 5 November 2026
receptive-field fits allowing onset or centre before the last lick
@author: Dinghao Luo
'''

#%% imports
from hashlib import sha1
from itertools import product

import numpy as np
from scipy.optimize import least_squares
from scipy.special import log_ndtr

#%% constants
BOUNDS = {
    'gaussian': (np.array([-1., .05]), np.array([7., 4.])),
    'decay': (np.array([-1., .1]), np.array([3., 10.])),
    'rise': (np.array([-1., .1]), np.array([3., 10.])),
    }
GRIDS = {
    'gaussian': (np.linspace(-1., 7., 15), np.geomspace(.05, 4., 12)),
    'decay': (np.linspace(-1., 3., 10), np.geomspace(.1, 10., 12)),
    'rise': (np.linspace(-1., 3., 10), np.geomspace(.1, 10., 12)),
    }
_GRID_CACHE = {}

#%% shapes
def field_shape(name, t, interval, params, rho=0., tbar=None):
    '''evaluate the field, scaling only the specified shape parameters'''
    mu, width = params
    scale = 1. if tbar is None else np.maximum(1. + rho * (interval - tbar) / tbar, .1)
    if name == 'gaussian':
        return np.exp(-.5 * ((t - mu * scale) / (width * scale)) ** 2)
    x = interval - t if name == 'rise' else t
    tau = width * scale
    z = (x - mu) / .2
    return np.exp(.5 * (.2 / tau) ** 2 - (x - mu) / tau + log_ndtr(z - .2 / tau))


def linear_amplitude(y, shape, weights=None):
    if weights is None:
        weights = np.ones(len(y))
    shape_mean = np.average(shape, weights=weights)
    y_mean = np.average(y, weights=weights)
    centred = shape - shape_mean
    denom = np.dot(weights, centred * centred)
    amplitude = max(float(np.dot(weights * centred, y - y_mean) / denom), 0.) if denom > 0 else 0.
    return float(y_mean - amplitude * shape_mean), amplitude


def compressed_observations(name, t, interval, y, rho=0.):
    '''weighted means at identical coordinates preserve the original residual minimiser'''
    if rho != 0:
        return t, interval, y, np.ones(len(y))
    x = interval - t if name == 'rise' else t
    unique, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    means = np.bincount(inverse, weights=y) / counts
    if name == 'rise':
        return np.zeros(len(unique)), unique, means, counts
    return unique, np.zeros(len(unique)), means, counts


def fit_field(name, t, interval, y, rho=0., tbar=None, initial=None):
    '''profile linear coefficients on the coarse grid and through nonlinear refinement'''
    if name == 'flat':
        return np.array([np.mean(y)]), True
    t, interval, y, weights = compressed_observations(name, t, interval, y, rho)
    key = None
    if rho == 0 and initial is None:
        key = name, sha1(t.tobytes() + interval.tobytes() + weights.tobytes()).digest()
    cached = _GRID_CACHE.get(key) if key is not None else None
    if cached is None:
        nonlinear = np.array(list(product(*GRIDS[name])) if initial is None else [initial])
        shapes = field_shape(name, t[None, :], interval[None, :], nonlinear.T[:, :, None], rho, tbar)
        means = np.average(shapes, axis=1, weights=weights)
        centred = shapes - means[:, None]
        denom = (centred ** 2) @ weights
        cached = nonlinear, means, centred, denom
        if key is not None:
            _GRID_CACHE[key] = cached
    nonlinear, means, centred, denom = cached
    y_mean = np.average(y, weights=weights)
    covariance = centred @ (weights * (y - y_mean))
    amplitude = np.maximum(np.divide(covariance, denom, out=np.zeros_like(covariance), where=denom > 0), 0.)
    score = amplitude * covariance
    best = nonlinear[np.argmax(score)]

    def residual(params):
        shape = field_shape(name, t, interval, params, rho, tbar)
        baseline, amplitude = linear_amplitude(y, shape, weights)
        return np.sqrt(weights) * (baseline + amplitude * shape - y)

    lower, upper = BOUNDS[name]
    result = least_squares(residual, best, bounds=(lower, upper), x_scale='jac')
    shape = field_shape(name, t, interval, result.x, rho, tbar)
    baseline, amplitude = linear_amplitude(y, shape, weights)
    return np.r_[baseline, amplitude, result.x], result.success


def predict(name, t, interval, params, rho=0., tbar=None):
    if name == 'flat':
        return np.full_like(t, params[0], dtype=float)
    return params[0] + params[1] * field_shape(name, t, interval, params[2:], rho, tbar)


def fit_scaling(name, t, interval, y, tbar):
    best_sse = np.inf
    best = None
    success = True
    for rho in np.round(np.arange(-1., 2.01, .1), 10):
        params, ok = fit_field(name, t, interval, y, float(rho), tbar)
        success &= ok
        residual = y - predict(name, t, interval, params, float(rho), tbar)
        sse = np.sum(residual ** 2)
        if sse < best_sse:
            best_sse = sse
            best = params, float(rho)
    return best[0], best[1], success
