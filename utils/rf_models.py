# -*- coding: utf-8 -*-
'''
Created on Thu Nov  5 11:24:40 2026

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
def field_shape(name, bin_times, interval, params, rho=0., tbar=None):
    mu, width = params
    scale = 1. if tbar is None else np.maximum(1. + rho * (interval - tbar) / tbar, .1)
    if name == 'gaussian':
        return np.exp(-.5 * ((bin_times - mu * scale) / (width * scale)) ** 2)
    coordinates = interval - bin_times if name == 'rise' else bin_times
    tau = width * scale
    onset_distance = (coordinates - mu) / .2
    return np.exp(.5 * (.2 / tau) ** 2 - (coordinates - mu) / tau + log_ndtr(onset_distance - .2 / tau))


def linear_amplitude(observed_values, shape, weights=None):
    if weights is None:
        weights = np.ones(len(observed_values))
    shape_mean = np.average(shape, weights=weights)
    observed_mean = np.average(observed_values, weights=weights)
    centred = shape - shape_mean
    denominator = np.dot(weights, centred * centred)
    amplitude = max(float(np.dot(weights * centred, observed_values - observed_mean) / denominator), 0.) if denominator > 0 else 0.
    return float(observed_mean - amplitude * shape_mean), amplitude


def compressed_observations(name, bin_times, interval, observed_values, rho=0.):
    '''weighted means at identical coordinates preserve the original residual minimiser'''
    if rho != 0:
        return bin_times, interval, observed_values, np.ones(len(observed_values))
    coordinates = interval - bin_times if name == 'rise' else bin_times
    unique, inverse, counts = np.unique(coordinates, return_inverse=True, return_counts=True)
    means = np.bincount(inverse, weights=observed_values) / counts
    if name == 'rise':
        return np.zeros(len(unique)), unique, means, counts
    return unique, np.zeros(len(unique)), means, counts


def fit_field(name, bin_times, interval, observed_values, rho=0., tbar=None, initial=None, equal_time_weight=False, bounds=None):
    '''profile linear coefficients on the coarse grid and through nonlinear refinement

    equal_time_weight gives compressed model-axis coordinates equal weight; the
    caller chooses their trial support, while bounds changes only the current fit
    '''
    if equal_time_weight and rho != 0:
        raise ValueError('equal time-bin weights require a single time axis')
    if name == 'flat':
        return np.array([np.mean(observed_values)]), True
    bin_times, interval, observed_values, weights = compressed_observations(name, bin_times, interval, observed_values, rho)
    if equal_time_weight and rho == 0:
        weights = np.ones(len(weights))
    lower, upper = BOUNDS[name] if bounds is None else bounds
    key = None
    if rho == 0 and initial is None:
        key = name, tuple(lower), tuple(upper), sha1(bin_times.tobytes() + interval.tobytes() + weights.tobytes()).digest()
    cached = _GRID_CACHE.get(key) if key is not None else None
    if cached is None:
        grid = GRIDS[name] if bounds is None else (np.linspace(lower[0], upper[0], len(GRIDS[name][0])),
                                                  np.geomspace(lower[1], upper[1], len(GRIDS[name][1])))
        nonlinear = np.array(list(product(*grid)) if initial is None else [initial])
        shapes = field_shape(name, bin_times[None, :], interval[None, :], nonlinear.T[:, :, None], rho, tbar)
        means = np.average(shapes, axis=1, weights=weights)
        centred = shapes - means[:, None]
        denominator = (centred ** 2) @ weights
        cached = nonlinear, means, centred, denominator
        if key is not None:
            _GRID_CACHE[key] = cached
    nonlinear, means, centred, denominator = cached
    observed_mean = np.average(observed_values, weights=weights)
    covariance = centred @ (weights * (observed_values - observed_mean))
    amplitude = np.maximum(np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0), 0.)
    score = amplitude * covariance
    best = nonlinear[np.argmax(score)]

    def residual(params):
        shape = field_shape(name, bin_times, interval, params, rho, tbar)
        baseline, amplitude = linear_amplitude(observed_values, shape, weights)
        return np.sqrt(weights) * (baseline + amplitude * shape - observed_values)

    result = least_squares(residual, best, bounds=(lower, upper), x_scale='jac')
    shape = field_shape(name, bin_times, interval, result.x, rho, tbar)
    baseline, amplitude = linear_amplitude(observed_values, shape, weights)
    return np.r_[baseline, amplitude, result.x], result.success


def predict(name, bin_times, interval, params, rho=0., tbar=None):
    if name == 'flat':
        return np.full_like(bin_times, params[0], dtype=float)
    return params[0] + params[1] * field_shape(name, bin_times, interval, params[2:], rho, tbar)


def fit_scaling(name, bin_times, interval, observed_values, tbar):
    best_sse = np.inf
    best = None
    success = True
    for rho in np.round(np.arange(-1., 2.01, .1), 10):
        params, fit_success = fit_field(name, bin_times, interval, observed_values, float(rho), tbar)
        success &= fit_success
        residual = observed_values - predict(name, bin_times, interval, params, float(rho), tbar)
        sse = np.sum(residual ** 2)
        if sse < best_sse:
            best_sse = sse
            best = params, float(rho)
    return best[0], best[1], success
