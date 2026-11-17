# -*- coding: utf-8 -*-
'''
Created on Mon Nov 16 13:12:22 2026

compare two-exponential bumps in the saved wider-window folds

@author: Dinghao Luo
'''

#%% imports
import os
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.rf_models import linear_amplitude
from utils.trial_error_scores import trial_mse

DATA   = ROOT / 'data' / 'binned_wider'
WIDER  = ROOT / 'outputs' / 'wider_window_fits'
OUTPUT = ROOT / 'outputs' / 'two_exponential'
GROUPS = ['tic_down', 'tic_up', 'time_cell']
LOWER  = np.array([0., .1, .1])
UPPER  = np.array([7., 10., 10.])
GRID   = np.array(list(product([0., .5, 1., 2., 3., 4., 5.5, 7.],
                               np.geomspace(.1, 10., 4), np.geomspace(.1, 10., 4))))

#%% bump and exact compression
def bump_shape(t, interval, params, anchoring, derivatives=False):
    t, interval = np.broadcast_arrays(t, interval)
    u, tau1, tau2 = params
    p = np.minimum(u, interval) if anchoring == 'last' else np.maximum(interval - u, 0.)
    dp = (u < interval).astype(float) * (1. if anchoring == 'last' else -1.)
    shape = np.zeros(t.shape)
    jac = np.zeros((t.size, 3))
    # endpoint baselines also define the otherwise ambiguous p=0 and p=T cases
    for rising, tau in [(True, tau1), (False, tau2)]:
        keep = (t > 0) & (t < interval) & ((t <= p) if rising else (t > p))
        x = t[keep] if rising else (interval - t)[keep]
        length = p[keep] if rising else (interval - p)[keep]
        numerator = -np.expm1(-x / tau)
        denominator = -np.expm1(-length / tau)
        shape[keep] = numerator / denominator
        if derivatives:
            length_derivative = -numerator * np.exp(-length / tau) / (tau * denominator ** 2)
            tau_derivative = (-x * np.exp(-x / tau) * denominator +
                              numerator * length * np.exp(-length / tau)) / (tau ** 2 * denominator ** 2)
            jac[keep.ravel(), 0] = length_derivative * dp[keep] * (1. if rising else -1.)
            jac[keep.ravel(), 1 if rising else 2] = tau_derivative
    return (shape, jac) if derivatives else shape


def predict_bump(t, interval, params, anchoring):
    return params[0] + params[1] * bump_shape(t, interval, params[2:], anchoring)


def compress_pairs(t, interval, y):
    coordinates, inverse, counts = np.unique(np.c_[t, interval], axis=0,
                                             return_inverse=True, return_counts=True)
    means = np.bincount(inverse, weights=y) / counts
    return coordinates[:, 0], coordinates[:, 1], means, counts


def coarse_grid(t, interval, weights, anchoring):
    shapes = np.array([bump_shape(t, interval, p, anchoring) for p in GRID])
    means = np.average(shapes, axis=1, weights=weights)
    centred = shapes - means[:, None]
    return centred, (centred ** 2) @ weights


def fit_bump(t, interval, y, weights, anchoring, grid=None):
    if grid is None:
        grid = coarse_grid(t, interval, weights, anchoring)
    centred, denominator = grid
    y_mean = np.average(y, weights=weights)
    covariance = centred @ (weights * (y - y_mean))
    scores = np.divide(np.maximum(covariance, 0.) ** 2, denominator,
                       out=np.zeros_like(covariance), where=denominator > 0)
    initial = GRID[np.argmax(scores)]
    square_root = np.sqrt(weights)
    previous, cached = None, None

    def evaluate(params):
        nonlocal previous, cached
        if previous is not None and np.array_equal(params, previous):
            return cached
        shape, derivative = bump_shape(t, interval, params, anchoring, True)
        shape_mean = np.average(shape, weights=weights)
        centred_shape = shape - shape_mean
        denom = np.dot(weights, centred_shape ** 2)
        cov = np.dot(weights * centred_shape, y - y_mean)
        amplitude = max(cov / denom, 0.) if denom > 0 else 0.
        derivative_mean = np.average(derivative, axis=0, weights=weights)
        amplitude_derivative = np.zeros(3)
        if amplitude > 0:
            dcov = derivative.T @ (weights * (y - y_mean))
            ddenom = 2. * derivative.T @ (weights * centred_shape)
            amplitude_derivative = (dcov * denom - cov * ddenom) / denom ** 2
        residual = square_root * (y_mean + amplitude * centred_shape - y)
        jacobian = square_root[:, None] * (amplitude * (derivative - derivative_mean) +
                                           centred_shape[:, None] * amplitude_derivative)
        previous = params.copy()
        cached = residual, jacobian
        return cached

    result = least_squares(lambda p: evaluate(p)[0], initial,
                           jac=lambda p: evaluate(p)[1], bounds=(LOWER, UPPER), x_scale='jac')
    shape = bump_shape(t, interval, result.x, anchoring)
    baseline, amplitude = linear_amplitude(y, shape, weights)
    return np.r_[baseline, amplitude, result.x], result.success, result.nfev


#%% held-out comparisons
def contrast_row(error_a, error_b, variance, bootstrap):
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError('saved residual variance must be finite and positive')
    values = (error_b - error_a) / (2. * variance)
    lower, upper = np.percentile(values[bootstrap].mean(axis=1), [2.5, 97.5])
    return dict(mean_contrast=values.mean(), lower=lower, upper=upper,
                decision='favour_a' if lower > 0 else 'favour_b' if upper < 0 else 'undecided')


def session_job(session):
    started = time.perf_counter()
    data = np.load(DATA / f'{session}.npz')
    signals, times, intervals, ids = [data[k] for k in ['dff', 't', 'T_k', 'neu_id']]
    fold_data = np.load(ROOT / 'outputs' / 'fits' / f'{session}_folds.npz')
    np.testing.assert_array_equal(data['trial_rows'], fold_data['trial_rows'])
    folds, bootstrap = fold_data['folds'], fold_data['bootstrap_trials']
    cells = pd.read_csv(WIDER / f'{session}_wider_parameters.csv').set_index('neu_id').loc[ids].reset_index()
    prior = pd.read_csv(WIDER / f'{session}_wider_contrasts.csv')
    assert prior.groupby('neu_id').flat_residual_variance.nunique().eq(1).all()
    variances = prior.groupby('neu_id').flat_residual_variance.first()
    failures, parameters, rows = [], [], []
    observed = np.isfinite(signals[0])
    if not np.array_equal(np.isfinite(signals), np.broadcast_to(observed, signals.shape)):
        raise ValueError(f'{session}: cell-specific observation masks need separate compression')
    with h5py.File(OUTPUT / f'{session}_oof.h5', 'w') as saved:
        saved.create_dataset('neu_id', data=ids)
        for anchoring in ['last', 'next']:
            held = np.full(signals.shape, np.nan)
            for stage in [0, 1, 2, 3, 4, 'full']:
                train = np.ones(len(intervals), dtype=bool) if stage == 'full' else folds != stage
                trial, bins = np.where(observed & train[:, None])
                coordinates, inverse, weights = np.unique(np.c_[times[bins], intervals[trial]], axis=0,
                                                          return_inverse=True, return_counts=True)
                t, interval = coordinates.T
                grid = coarse_grid(t, interval, weights, anchoring)
                test_trial, test_bins = np.where(observed & ~train[:, None])
                for index, cell in enumerate(cells.itertuples()):
                    y = np.bincount(inverse, weights=signals[index, trial, bins]) / weights
                    try:
                        params, ok, nfev = fit_bump(t, interval, y, weights, anchoring, grid)
                    except Exception as error:
                        failures.append(dict(session=session, neu_id=int(cell.neu_id), anchoring=anchoring,
                                             stage=str(stage), issue=repr(error)))
                        raise
                    if not ok:
                        failures.append(dict(session=session, neu_id=int(cell.neu_id), anchoring=anchoring,
                                             stage=str(stage), issue='least_squares did not converge'))
                    if stage == 'full':
                        row = dict(session=session, neu_id=int(cell.neu_id), group=cell.group,
                                   anchoring=anchoring, fit_success=bool(ok), nfev=nfev)
                        row.update(zip(['a', 'B', 'u', 'tau1', 'tau2'], params))
                        row['B_at_lower'] = params[1] <= 1e-5
                        for key, value, lo, hi in zip(['u', 'tau1', 'tau2'], params[2:], LOWER, UPPER):
                            row[f'{key}_at_lower'] = bool(np.isclose(value, lo, rtol=0, atol=1e-5))
                            row[f'{key}_at_upper'] = bool(np.isclose(value, hi, rtol=0, atol=1e-5))
                        parameters.append(row)
                    else:
                        held[index, test_trial, test_bins] = predict_bump(times[test_bins], intervals[test_trial], params, anchoring)
                print(session, anchoring, stage, 'seconds', round(time.perf_counter() - started, 1), flush=True)
            saved.create_dataset(anchoring, data=held, compression='lzf')
        with h5py.File(WIDER / f'{session}_wider_oof.h5', 'r') as original:
            np.testing.assert_array_equal(original['neu_id'][:], ids)
            np.testing.assert_array_equal(original['trial_rows'][:], data['trial_rows'] + 1)
            np.testing.assert_array_equal(original['t'][:], times)
            for index, cell in enumerate(cells.itertuples()):
                exponential = 'rise' if cell.group == 'tic_up' else 'decay'
                predictions = {name: saved[name][index] for name in ['last', 'next']}
                predictions.update({name: original[name][index] for name in ['gaussian', exponential]})
                for scoring in ['all', 'post_last_lick']:
                    signal = signals[index].astype(float).copy()
                    if scoring != 'all':
                        signal[:, times < 0] = np.nan
                    errors = {name: trial_mse(signal, fitted) for name, fitted in predictions.items()}
                    pairs = [(anchor, reference) for anchor in ['last', 'next'] for reference in ['gaussian', exponential]]
                    pairs.append(('last', 'next'))
                    for a, b in pairs:
                        rows.append(dict(session=session, neu_id=int(cell.neu_id), group=cell.group,
                                         model_a=a, model_b=b, scoring=scoring,
                                         **contrast_row(errors[a], errors[b], variances.loc[cell.neu_id], bootstrap)))
    pd.DataFrame(parameters).to_csv(OUTPUT / f'{session}_parameters.csv', index=False)
    pd.DataFrame(rows).to_csv(OUTPUT / f'{session}_contrasts.csv', index=False)
    status = dict(session=session, seconds=time.perf_counter() - started, failures=failures, cells=len(ids))
    (OUTPUT / f'{session}_status.json').write_text(json.dumps(status, indent=2))
    return status


def signed_test(values):
    values = np.asarray(values)
    return float(wilcoxon(values).pvalue) if np.any(values != 0) else 1.


def markdown(frame):
    def fmt(value):
        return f'{value:.5g}' if isinstance(value, (float, np.floating)) else str(value)
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |',
                      '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |',
                      *['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)]])


def summarise(seconds):
    parameters = pd.concat([pd.read_csv(p) for p in sorted(OUTPUT.glob('*_parameters.csv'))], ignore_index=True)
    cells = pd.concat([pd.read_csv(p) for p in sorted(OUTPUT.glob('*_contrasts.csv'))], ignore_index=True)
    parameters.to_csv(OUTPUT / 'parameters.csv', index=False)
    cells.to_csv(OUTPUT / 'contrasts.csv', index=False)
    rows = []
    for (group, a, b, scoring), part in cells.groupby(['group', 'model_a', 'model_b', 'scoring']):
        medians = part.groupby('session').mean_contrast.median()
        rows.append(dict(group=group, model_a=a, model_b=b, scoring=scoring, n=len(part), sessions=len(medians),
                         better=100 * part.decision.eq('favour_a').mean(), worse=100 * part.decision.eq('favour_b').mean(),
                         undecided=100 * part.decision.eq('undecided').mean(),
                         median_session_contrast=medians.median(), wilcoxon_p=signed_test(medians),
                         favoured=a if medians.median() > 0 else b if medians.median() < 0 else 'neither'))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT / 'comparison_summary.csv', index=False)
    rows = []
    for (group, anchor), part in parameters.groupby(['group', 'anchoring']):
        for key in ['u', 'tau1', 'tau2']:
            rows.append(dict(group=group, anchoring=anchor, parameter=key, n=len(part),
                             median=part[key].median(), q25=part[key].quantile(.25), q75=part[key].quantile(.75),
                             lower_bound_pct=100 * part[f'{key}_at_lower'].mean(),
                             upper_bound_pct=100 * part[f'{key}_at_upper'].mean()))
    parameter_summary = pd.DataFrame(rows)
    parameter_summary.to_csv(OUTPUT / 'parameter_summary.csv', index=False)
    statuses = [json.loads(p.read_text()) for p in OUTPUT.glob('*_status.json')]
    failures = [failure for status in statuses for failure in status['failures']]
    pd.DataFrame(failures, columns=['session', 'neu_id', 'anchoring', 'stage', 'issue']).to_csv(OUTPUT / 'failures.csv', index=False)
    text = '# two-exponential bumps\n\nAll observed trial bins enter the fit, with identical (t, T) pairs compressed to their means and weighted by count; the coarse grid profiles a and non-negative B before bounded least squares. The saved five folds, trial bootstrap draws and flat residual variance are retained, and t >= 0 is scored without refitting. Positive contrast favours model_a; the last-versus-next row is (error_next minus error_last) / (2 variance). All Wilcoxon tests use the 15 session medians and raw two-sided p-values.\n\n'
    text += markdown(comparison.drop(columns='favoured')) + '\n\n' + markdown(parameter_summary)
    text += '\n\nEndpoint values are defined as baseline even when the peak coincides with an endpoint; this is the continuous interior formula with a baseline endpoint convention, since the stated peak and baseline constraints conflict at p=0 or p=T. B has lower bound zero and no upper bound. Bound flags use absolute tolerance 1e-5.\n\n'
    for row in comparison[comparison.model_b.eq('next') & comparison.scoring.eq('all')].itertuples():
        text += f'{row.group}: session medians favour {row.favoured} anchoring (p={row.wilcoxon_p:.5g}); this is ' + ('significant' if row.wilcoxon_p < .05 else 'not significant') + '.\n\n'
    if (OUTPUT / 'example_objective_check.csv').exists():
        text += 'After seeing the low time-cell R2, the 12 saved example fits were checked against their best coarse-grid objective without changing any fit; every refined objective was at least as good within 6e-9 SSE. The prescribed falling branch has negative curvature for positive B, whereas an ordinary decaying exponential has positive curvature. This check does not establish a global optimum; details are in example_objective_check.csv.\n\n'
    text += f'Fit failures: {len(failures)} of {len(parameters) * 6} fits. Wall time: {seconds:.1f} s. Parameter IQRs describe cells, while trial-bootstrap intervals describe each cell\'s held-out contrast. Local minima remain the weakest step; both anchorings use the same coarse grid and bounded refinement.\n'
    (OUTPUT / 'summary.md').write_text(text, encoding='utf-8')
    (OUTPUT / 'run_report.json').write_text(json.dumps(dict(seconds=seconds, failures=len(failures), fits=len(parameters)*6), indent=2))
    print(text, flush=True)


def main():
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sessions = sorted(p.name.replace('_wider_parameters.csv', '') for p in WIDER.glob('*_wider_parameters.csv'))
    with ProcessPoolExecutor(max_workers=6) as pool:
        list(pool.map(session_job, sessions))
    summarise(time.perf_counter() - started)


if __name__ == '__main__':
    main()
