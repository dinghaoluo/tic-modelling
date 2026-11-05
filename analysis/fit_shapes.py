'''Created on 4 November 2026
trial-held-out shape comparison and interval scaling
@author: Dinghao Luo
'''

#%% imports
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rec_list import SESSIONS_4S
from utils.rf_models import BOUNDS, fit_field, predict

#%% constants
OUTPUT = ROOT / 'outputs' / 'fits_v1'
SHAPES = ['gaussian', 'decay', 'rise']
MODELS = ['flat', *SHAPES]
SEED = 20261104
RHO_GRID = np.arange(-10, 21) / 10

#%% scores
def bootstrap_difference(difference, resamples):
    totals = difference[resamples].sum(axis=1)
    return float(difference.sum()), *np.percentile(totals, [2.5, 97.5]).tolist()


def trial_scores(y, prediction, trial, n_trials, noise_sd):
    residual = y - prediction
    loglik = -.5 * (np.log(2 * np.pi * noise_sd ** 2) + (residual / noise_sd) ** 2)
    ll = np.bincount(trial, weights=loglik, minlength=n_trials)
    sse = np.bincount(trial, weights=residual ** 2, minlength=n_trials)
    count = np.bincount(trial, minlength=n_trials)
    total = np.bincount(trial, weights=y, minlength=n_trials)
    total_sq = np.bincount(trial, weights=y ** 2, minlength=n_trials)
    sst = total_sq - np.divide(total ** 2, count, out=np.zeros(n_trials), where=count > 0)
    r2 = np.full(n_trials, np.nan)
    np.divide(sse, sst, out=r2, where=sst > 0)
    r2 = 1 - r2
    r2[count == 0] = np.nan
    return ll, r2


def cv_model(name, y, t, interval, trial, folds, n_trials, tbar, scaling=False):
    held_prediction = np.full_like(y, np.nan)
    held_ll = np.zeros(n_trials)
    held_r2 = np.full(n_trials, np.nan)
    success = True
    for fold in range(5):
        train = folds[trial] != fold
        test = ~train
        rho = 0.
        if scaling:
            params, rho, ok = fit_scaling(name, t[train], interval[train], y[train], tbar)
        else:
            params, ok = fit_field(name, t[train], interval[train], y[train])
        success &= ok
        training_residual = y[train] - predict(name, t[train], interval[train], params, rho, tbar)
        noise_sd = np.sqrt(np.mean(training_residual ** 2))
        if not np.isfinite(noise_sd) or noise_sd <= 0:
            raise ValueError('non-positive or non-finite training noise SD')
        prediction = predict(name, t[test], interval[test], params, rho, tbar)
        held_prediction[test] = prediction
        ll, r2 = trial_scores(y[test], prediction, trial[test], n_trials, noise_sd)
        held_ll[folds == fold] = ll[folds == fold]
        held_r2[folds == fold] = r2[folds == fold]
    denominator = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - np.sum((y - held_prediction) ** 2) / denominator if denominator > 0 else np.nan
    return held_ll, held_r2, r2, success


def fit_scaling(name, t, interval, y, tbar):
    best_sse = np.inf
    best = None
    success = True
    for rho in RHO_GRID:
        params, ok = fit_field(name, t, interval, y, float(rho), tbar)
        success &= ok
        residual = y - predict(name, t, interval, params, float(rho), tbar)
        sse = np.sum(residual ** 2)
        if sse < best_sse:
            best_sse = sse
            best = params, float(rho)
    return best[0], best[1], success


def polynomial_cv(degree, y, t, trial, folds, n_trials):
    ll = np.zeros(n_trials)
    r2_trial = np.full(n_trials, np.nan)
    held = np.empty_like(y)
    design = np.polynomial.polynomial.polyvander(t / 7, degree)
    for fold in range(5):
        train = folds[trial] != fold
        test = ~train
        beta = np.linalg.lstsq(design[train], y[train], rcond=None)[0]
        noise = np.sqrt(np.mean((y[train] - design[train] @ beta) ** 2))
        prediction = design[test] @ beta
        held[test] = prediction
        fold_ll, fold_r2 = trial_scores(y[test], prediction, trial[test], n_trials, noise)
        ll[folds == fold] = fold_ll[folds == fold]
        r2_trial[folds == fold] = fold_r2[folds == fold]
    sst = np.sum((y - y.mean()) ** 2)
    return ll, r2_trial, 1 - np.sum((y - held) ** 2) / sst if sst > 0 else np.nan


#%% sessions
def fit_session(session):
    started = time.perf_counter()
    data = np.load(ROOT / 'data' / 'binned' / f'{session}.npz')
    dff, intervals, times = data['dff'], data['T_k'], data['t']
    labels = pd.read_csv(ROOT / 'data' / 'labels.csv').query('session == @session')
    n_trials = len(intervals)
    if n_trials < 5:
        raise ValueError(f'{session}: fewer trials than cross-validation folds')
    rng = np.random.default_rng(SEED + SESSIONS_4S.index(session))
    folds = np.empty(n_trials, dtype=int)
    for fold, indices in enumerate(np.array_split(rng.permutation(n_trials), 5)):
        folds[indices] = fold
    resamples = rng.integers(0, n_trials, size=(1000, n_trials))
    np.savez(OUTPUT / f'{session}_folds.npz', trial_rows=data['trial_rows'], folds=folds)
    rows, trial_rows, failures = [], [], []
    for neuron, signal in enumerate(dff):
        trial, bin_idx = np.where(np.isfinite(signal))
        y = signal[trial, bin_idx].astype(float)
        t, interval = times[bin_idx], intervals[trial]
        row = labels.iloc[neuron].to_dict()
        row['fit_success'] = True
        model_ll = {}
        for name in MODELS:
            ll, r2_trial, r2, ok = cv_model(name, y, t, interval, trial, folds, n_trials, intervals.mean())
            model_ll[name] = ll
            row[f'{name}_held_ll'] = float(ll.sum())
            row[f'{name}_held_r2'] = r2
            row['fit_success'] &= ok
            for k in range(n_trials):
                trial_rows.append(dict(neu_id=neuron + 1, trial_row=int(data['trial_rows'][k]) + 1,
                                       model=name, held_ll=ll[k], held_r2=r2_trial[k]))
        ordered = sorted(SHAPES, key=lambda name: row[f'{name}_held_ll'], reverse=True)
        best, runner_up = ordered[:2]
        row['best_shape'] = best
        row['next_shape'] = runner_up
        for label, other in [('flat', 'flat'), ('next', runner_up)]:
            gain, lower, upper = bootstrap_difference(model_ll[best] - model_ll[other], resamples)
            row[f'll_gain_{label}'] = gain
            row[f'll_gain_{label}_lower'] = lower
            row[f'll_gain_{label}_upper'] = upper
        row['beats_flat'] = row['ll_gain_flat_lower'] > 0
        row['preferred'] = ('flat' if not row['beats_flat'] else
                            ('undecided' if row['ll_gain_next_lower'] <= 0 <= row['ll_gain_next_upper'] else best))
        row['best_shape_held_r2'] = row[f'{best}_held_r2']
        for name in SHAPES:
            params, ok = fit_field(name, t, interval, y)
            row['fit_success'] &= ok
            names = ['a0', 'a', 'mu', 'sigma' if name == 'gaussian' else 'tau']
            for param_name, value in zip(names, params):
                row[f'{name}_{param_name}'] = value
            lower, upper = BOUNDS[name]
            at_bound = np.isclose(params[2:], lower, atol=1e-5, rtol=0) | np.isclose(params[2:], upper, atol=1e-5, rtol=0)
            for param_name, value in zip(names[2:], at_bound):
                row[f'{name}_{param_name}_at_bound'] = bool(value)
            row[f'{name}_a_at_bound'] = bool(params[1] <= 1e-5)
        for degree in range(1, 5):
            ll, r2_trial, r2 = polynomial_cv(degree, y, t, trial, folds, n_trials)
            row[f'poly{degree}_held_ll'] = float(ll.sum())
            row[f'poly{degree}_held_r2'] = r2
            for k in range(n_trials):
                trial_rows.append(dict(neu_id=neuron + 1, trial_row=int(data['trial_rows'][k]) + 1,
                                       model=f'poly{degree}', held_ll=ll[k], held_r2=r2_trial[k]))
        row['best_poly_degree'] = max(range(1, 5), key=lambda degree: row[f'poly{degree}_held_r2'])
        best_degree = row['best_poly_degree']
        row['best_poly_held_r2'] = row[f'poly{best_degree}_held_r2']
        if row['preferred'] in SHAPES:
            name = row['preferred']
            params, rho, ok = fit_scaling(name, t, interval, y, intervals.mean())
            row['fit_success'] &= ok
            row['rho'] = rho
            row['rho_at_bound'] = rho in (RHO_GRID[0], RHO_GRID[-1])
            lower_bounds, upper_bounds = BOUNDS[name]
            scaled_bounds = (np.isclose(params[2:], lower_bounds, atol=1e-5, rtol=0)
                             | np.isclose(params[2:], upper_bounds, atol=1e-5, rtol=0))
            row['scaled_mu_at_bound'] = bool(scaled_bounds[0])
            row['scaled_width_at_bound'] = bool(scaled_bounds[1])
            row['scaled_a_at_bound'] = bool(params[1] <= 1e-5)
            for param_name, value in zip(['a0', 'a', 'mu', 'sigma' if name == 'gaussian' else 'tau'], params):
                row[f'scaled_{param_name}'] = value
            ll, r2_trial, r2, ok = cv_model(name, y, t, interval, trial, folds, n_trials, intervals.mean(), scaling=True)
            row['fit_success'] &= ok
            row['scaled_held_r2'] = r2
            gain, lower, upper = bootstrap_difference(ll - model_ll[name], resamples)
            row.update(scaling_ll_gain=gain, scaling_ll_gain_lower=lower, scaling_ll_gain_upper=upper)
            for k in range(n_trials):
                trial_rows.append(dict(neu_id=neuron + 1, trial_row=int(data['trial_rows'][k]) + 1,
                                       model='scaled', held_ll=ll[k], held_r2=r2_trial[k]))
        if not row['fit_success']:
            failures.append(neuron + 1)
        rows.append(row)
        if len(rows) % 10 == 0:
            pd.DataFrame(rows).to_csv(OUTPUT / f'{session}.csv', index=False)
            print(session, 'completed neuron', neuron + 1, flush=True)
    pd.DataFrame(rows).to_csv(OUTPUT / f'{session}.csv', index=False)
    pd.DataFrame(trial_rows).to_csv(OUTPUT / f'{session}_held_trials.csv', index=False)
    elapsed = time.perf_counter() - started
    (OUTPUT / f'{session}_timing.json').write_text(json.dumps(dict(seconds=elapsed, failed_neurons=failures), indent=2))
    print(session, 'complete in seconds', elapsed, flush=True)
    return session, elapsed, len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if not json.loads((OUTPUT / 'dataset_status.json').read_text())['alignment_passed']:
        raise ValueError('alignment gate did not pass')
    started = time.perf_counter()
    first, seconds, neurons = fit_session(SESSIONS_4S[0])
    labels = pd.read_csv(ROOT / 'data' / 'labels.csv')
    estimate = seconds * (len(labels) - neurons) / neurons / args.workers
    print('estimated remaining fitting hours after first session:', estimate / 3600, flush=True)
    if args.workers == 1:
        for session in SESSIONS_4S[1:]:
            fit_session(session)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(fit_session, SESSIONS_4S[1:]))
    combined = pd.concat([pd.read_csv(OUTPUT / f'{session}.csv') for session in SESSIONS_4S], ignore_index=True)
    combined.to_csv(OUTPUT / 'fits_all.csv', index=False)
    (OUTPUT / 'fit_timing.json').write_text(json.dumps(dict(seconds=time.perf_counter() - started), indent=2))


if __name__ == '__main__':
    main()


