'''
2026-11-07
fit a held-out ex-gaussian bridge to the wider-window traces
@author: Dinghao Luo
'''

#%% imports
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import log_ndtr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'binned_v1_3'
FITS = ROOT / 'outputs' / 'fits_v1_2'
WIDER = ROOT / 'outputs' / 'undecided_check'
OUTPUT = ROOT / 'outputs' / 'wider_bridge'
BOUNDS = (np.array([-1., .05, .05]), np.array([7., 4., 10.]))
GRID = (np.linspace(-1., 7., 5), np.geomspace(.05, 4., 3), np.geomspace(.05, 10., 4))
SEED = 20261107
LIMIT = None
UND_ONLY = False


def bridge_shape(t, interval, params, future=False):
    mu, sigma, tau = params
    x = interval - t if future else t
    log_pdf = (-np.log(2. * tau) + .5 * (sigma / tau) ** 2 - (x - mu) / tau +
               log_ndtr((x - mu) / sigma - sigma / tau))
    log_pdf -= np.nanmax(log_pdf, axis=0, keepdims=True) if log_pdf.ndim > 1 else np.nanmax(log_pdf)
    return np.exp(np.clip(log_pdf, -80., 20.))


def linear_amplitude(y, shape):
    shape_mean = shape.mean()
    y_mean = y.mean()
    centred = shape - shape_mean
    denominator = np.dot(centred, centred)
    amplitude = max(float(np.dot(centred, y - y_mean) / denominator), 0.) if denominator else 0.
    return float(y_mean - amplitude * shape_mean), amplitude


def fit_bridge(t, interval, y, future=False):
    candidates = np.array(np.meshgrid(*GRID)).reshape(3, -1).T
    sample = np.linspace(0, len(y) - 1, min(len(y), 2500)).astype(int)
    sampled_shapes = np.array([bridge_shape(t[sample], interval[sample], params, future) for params in candidates])
    means = sampled_shapes.mean(axis=1)
    centred = sampled_shapes - means[:, None]
    covariance = centred @ (y[sample] - y[sample].mean())
    denominator = (centred ** 2).sum(axis=1)
    score = np.maximum(covariance, 0.) ** 2 / np.maximum(denominator, 1e-12)
    initial = candidates[np.argmax(score)]

    fit_idx = np.linspace(0, len(y) - 1, min(len(y), 5000)).astype(int)

    def residual(params):
        shape = bridge_shape(t[fit_idx], interval[fit_idx], params, future)
        baseline, amplitude = linear_amplitude(y[fit_idx], shape)
        return baseline + amplitude * shape - y[fit_idx]

    result = least_squares(residual, initial, bounds=BOUNDS, x_scale='jac', max_nfev=120)
    shape = bridge_shape(t, interval, result.x)
    baseline, amplitude = linear_amplitude(y, shape)
    return np.r_[baseline, amplitude, result.x], result.success


def predict(t, interval, params, future=False):
    shape = bridge_shape(t, interval, params[2:], future)
    return params[0] + params[1] * shape


def trial_mse(observed, prediction):
    return np.nanmean((observed - prediction) ** 2, axis=1)


def trial_contrast(error_a, error_b, variance):
    return (error_b - error_a) / (2. * variance)


def bootstrap_support(values, bootstrap):
    means = values[bootstrap].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(np.mean(means > 0.))


def fit_cv(signal, times, intervals, folds, future=False):
    held = np.full(signal.shape, np.nan)
    failures = []
    for fold in range(5):
        train = folds != fold
        test = ~train
        trial, bins = np.where(np.isfinite(signal) & train[:, None])
        y = signal[trial, bins].astype(float)
        params, ok = fit_bridge(times[bins], intervals[trial], y, future)
        test_trial, test_bins = np.where(np.isfinite(signal) & test[:, None])
        held[test_trial, test_bins] = predict(times[test_bins], intervals[test_trial], params, future)
        if not ok:
            failures.append(fold)
    return held, failures


def session_job(session):
    data = np.load(DATA / f'{session}.npz')
    folds = np.load(FITS / f'{session}_folds.npz')
    parameters = pd.read_csv(WIDER / f'{session}_wider_parameters.csv')
    contrasts = pd.read_csv(WIDER / f'{session}_wider_contrasts.csv')
    rows = []
    selected = parameters[parameters.group.isin(['tic_down', 'tic_up', 'time_cell'])].reset_index(drop=True)
    if UND_ONLY:
        undecided = contrasts.loc[contrasts.primary & contrasts.decision.eq('undecided'), ['neu_id']]
        selected = selected[selected.neu_id.isin(undecided.neu_id)].reset_index(drop=True)
    if LIMIT is not None:
        selected = selected.groupby('group', group_keys=False).head(LIMIT).reset_index(drop=True)
    neu_to_row = {int(neu): i for i, neu in enumerate(parameters.neu_id)}
    with h5py.File(WIDER / f'{session}_wider_oof.h5', 'r') as saved:
        for row in selected.itertuples(index=False):
            index = neu_to_row[int(row.neu_id)]
            signal = data['dff'][index]
            bridge, failures = fit_cv(signal, data['t'], data['T_k'], folds['folds'], row.group == 'tic_up')
            bridge_error = trial_mse(signal, bridge)
            expected = 'rise' if row.group == 'tic_up' else 'decay'
            gaussian = saved['gaussian'][index]
            exponential = saved[expected][index]
            gaussian_error = trial_mse(signal, gaussian)
            exponential_error = trial_mse(signal, exponential)
            variance = float(contrasts.loc[(contrasts.neu_id == row.neu_id) &
                                           (contrasts.contrast == f'gaussian_minus_{expected}'),
                                           'flat_residual_variance'].iloc[0])
            bridge_gaussian = trial_contrast(bridge_error, gaussian_error, variance)
            bridge_exponential = trial_contrast(bridge_error, exponential_error, variance)
            bg_lower, bg_upper, bg_support = bootstrap_support(bridge_gaussian, folds['bootstrap_trials'])
            be_lower, be_upper, be_support = bootstrap_support(bridge_exponential, folds['bootstrap_trials'])
            rows.append(dict(session=session, neu_id=int(row.neu_id), group=row.group,
                             bridge_held_mse=float(bridge_error.mean()),
                             gaussian_held_mse=float(gaussian_error.mean()),
                             exponential_held_mse=float(exponential_error.mean()),
                             bridge_minus_gaussian=float(bridge_gaussian.mean()),
                             bridge_minus_exponential=float(bridge_exponential.mean()),
                             bridge_gaussian_lower=bg_lower, bridge_gaussian_upper=bg_upper,
                             bridge_gaussian_support=bg_support,
                             bridge_exponential_lower=be_lower, bridge_exponential_upper=be_upper,
                             bridge_exponential_support=be_support,
                             bridge_decisive=bool(bg_lower > 0 and be_lower > 0),
                             failures=';'.join(map(str, failures))))
    return rows


def init_worker(limit, undecided_only):
    global LIMIT, UND_ONLY
    LIMIT = limit
    UND_ONLY = undecided_only


def main():
    global LIMIT, UND_ONLY
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--sessions', type=int, default=None)
    parser.add_argument('--undecided', action='store_true')
    args = parser.parse_args()
    LIMIT = args.limit
    UND_ONLY = args.undecided
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sessions = sorted(path.name.replace('_wider_parameters.csv', '')
                      for path in WIDER.glob('*_wider_parameters.csv'))
    if args.sessions is not None:
        sessions = sessions[:args.sessions]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(sessions)),
                             initializer=init_worker, initargs=(LIMIT, UND_ONLY)) as pool:
        results = list(pool.map(session_job, sessions))
    frame = pd.DataFrame([row for result in results for row in result])
    if LIMIT:
        output_name = 'bridge_pilot.csv'
    elif UND_ONLY:
        output_name = 'bridge_undecided.csv'
    else:
        output_name = 'bridge_all.csv'
    frame.to_csv(OUTPUT / output_name, index=False)
    print(frame.groupby('group')[['bridge_minus_gaussian', 'bridge_minus_exponential']].agg(['median', 'count']).to_string())
    print('failures:', (frame.failures != '').sum(), 'cells:', len(frame), 'limit:', LIMIT)


if __name__ == '__main__':
    main()
