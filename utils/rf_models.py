'''Created on 4 November 2026
bounded receptive-field shapes and least-squares fitting
@author: Dinghao Luo
'''

#%% imports
from itertools import product

import numpy as np
from scipy.optimize import least_squares
from scipy.special import log_ndtr

#%% constants
BOUNDS = {
    'gaussian': (np.array([0., .05]), np.array([7., 4.])),
    'decay': (np.array([0., .1]), np.array([3., 10.])),
    'rise': (np.array([0., .1]), np.array([3., 10.])),
    }
GRIDS = {
    'gaussian': (np.linspace(0., 7., 15), np.geomspace(.05, 4., 12)),
    'decay': (np.linspace(0., 3., 10), np.geomspace(.1, 10., 12)),
    'rise': (np.linspace(0., 3., 10), np.geomspace(.1, 10., 12)),
    }

#%% shapes
def field_shape(name, t, interval, params, rho=0., tbar=None):
    '''evaluate the field, with trial scaling applied only to the specified parameters'''
    mu, width = params
    scale = 1. if tbar is None else np.maximum(1. + rho * (interval - tbar) / tbar, .1)
    if name == 'gaussian':
        return np.exp(-.5 * ((t - mu * scale) / (width * scale)) ** 2)
    x = interval - t if name == 'rise' else t
    tau = width * scale
    z = (x - mu) / .2
    # log normal cdf avoids overflow in the product of erfc and the exponential
    return np.exp(.5 * (.2 / tau) ** 2 - (x - mu) / tau + log_ndtr(z - .2 / tau))


def linear_amplitude(y, shape):
    centred = shape - np.mean(shape)
    denom = np.dot(centred, centred)
    amplitude = max(float(np.dot(centred, y - np.mean(y)) / denom), 0.) if denom > 0 else 0.
    baseline = float(np.mean(y) - amplitude * np.mean(shape))
    return baseline, amplitude


def fit_field(name, t, interval, y, rho=0., tbar=None, initial=None):
    '''profile the non-negative amplitude on a coarse grid, then refine its best point'''
    if name == 'flat':
        return np.array([np.mean(y)]), True
    best = None
    best_sse = np.inf
    candidates = product(*GRIDS[name]) if initial is None else [initial]
    for nonlinear in candidates:
        shape = field_shape(name, t, interval, nonlinear, rho, tbar)
        a0, a = linear_amplitude(y, shape)
        residual = a0 + a * shape - y
        sse = np.dot(residual, residual)
        if sse < best_sse:
            best_sse = sse
            best = np.array([a0, a, *nonlinear])
    lower, upper = BOUNDS[name]
    result = least_squares(
        lambda p: p[0] + p[1] * field_shape(name, t, interval, p[2:], rho, tbar) - y,
        best, bounds=(np.r_[-np.inf, 0., lower], np.r_[np.inf, np.inf, upper]),
        x_scale='jac',
        )
    return result.x, result.success


def predict(name, t, interval, params, rho=0., tbar=None):
    if name == 'flat':
        return np.full_like(t, params[0], dtype=float)
    return params[0] + params[1] * field_shape(name, t, interval, params[2:], rho, tbar)
