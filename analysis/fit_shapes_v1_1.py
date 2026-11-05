'''Created on 5 November 2026
fit single-trial fields and compare their out-of-fold mean traces
@author: Dinghao Luo
'''

#%% imports
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rec_list import SESSIONS_4S
from utils.rf_models import BOUNDS, fit_field, predict
from utils.mean_trace_scores import mean_trace_scores, resample_weights
from analysis.fit_shapes import fit_scaling

#%% constants
OUTPUT = ROOT / 'outputs' / 'fits_v1_1'
DATA = ROOT / 'data' / 'binned_v1_1'
SHAPES = ['gaussian', 'decay', 'rise']
MODELS = ['flat', *SHAPES, 'poly1', 'poly2', 'poly3', 'poly4']
SEED = 20261104

#%% cross-validation
def cv_prediction(name, signal, times, intervals, folds, scaling=False):
    '''fit observed single-trial bins, never a trial-averaged training trace'''
    trial, bins = np.where(np.isfinite(signal))
    y, t, interval = signal[trial, bins].astype(float), times[bins], intervals[trial]
    held = np.full(signal.shape, np.nan)
    failures = []
    for fold in range(5):
        train = folds[trial] != fold
        test = ~train
        if name.startswith('poly'):
            degree = int(name[-1])
            design = np.polynomial.polynomial.polyvander(t / 7, degree)
            params = np.linalg.lstsq(design[train], y[train], rcond=None)[0]
            prediction = design[test] @ params
            ok = True
        else:
            rho = 0.
            if scaling:
                params, rho, ok = fit_scaling(name, t[train], interval[train], y[train], intervals.mean())
            else:
                params, ok = fit_field(name, t[train], interval[train], y[train])
            prediction = predict(name, t[test], interval[test], params, rho, intervals.mean())
        held[trial[test], bins[test]] = prediction
        if not ok:
            failures.append(dict(model=name, fold=fold, stage='scaled_cv' if scaling else 'cv'))
    return held, failures


def score_prediction(signal, prediction, weights):
    ll, r2, bins = mean_trace_scores(signal, prediction)
    boot, _, _ = mean_trace_scores(signal, prediction, weights)
    return float(ll[0]), float(r2[0]), int(bins[0]), boot


def interval_difference(point, reference, boot, boot_reference):
    delta = boot - boot_reference
    if not np.isfinite(delta).all():
        raise ValueError('non-finite bootstrap likelihood difference; cannot substitute another score')
    return dict(gain=point - reference, lower=float(np.percentile(delta, 2.5)),
                upper=float(np.percentile(delta, 97.5)))


#%% session fits
def fit_session(session):
    started = time.perf_counter()
    data = np.load(DATA / f'{session}.npz')
    dff, intervals, times = data['dff'], data['T_k'], data['t']
    labels = pd.read_csv(OUTPUT / 'labels.csv').query('session == @session').reset_index(drop=True)
    n_trials = len(intervals)
    if n_trials < 10:
        raise ValueError(f'{session}: fewer than ten kept trials')
    rng = np.random.default_rng(SEED + SESSIONS_4S.index(session))
    folds = np.empty(n_trials, dtype=int)
    for fold, indices in enumerate(np.array_split(rng.permutation(n_trials), 5)):
        folds[indices] = fold
    resamples = rng.integers(0, n_trials, size=(1000, n_trials))
    weights = resample_weights(resamples, n_trials)
    np.savez(OUTPUT / f'{session}_folds.npz', trial_rows=data['trial_rows'], folds=folds,
             bootstrap_trials=resamples)
    rows, failures = [], []
    with h5py.File(OUTPUT / f'{session}_oof.h5', 'w') as saved:
        saved.create_dataset('trial_rows', data=data['trial_rows'] + 1)
        saved.create_dataset('t', data=times)
        saved.attrs['description'] = 'out-of-fold predictions; neuron, trial, bin; NaN wherever dFFGF is unobserved'
        for name in [*MODELS, 'scaled']:
            saved.create_dataset(name, shape=dff.shape, dtype='f8', chunks=(1, n_trials, len(times)),
                                 compression='lzf', fillvalue=np.nan)
        for neuron, signal in enumerate(dff):
            row = labels.iloc[neuron].to_dict()
            bootstrap_ll = {}
            cell_failures = []
            for name in MODELS:
                held, failed = cv_prediction(name, signal, times, intervals, folds)
                saved[name][neuron] = held
                cell_failures.extend(failed)
                ll, r2, bins, boot = score_prediction(signal, held, weights)
                row[f'{name}_held_ll'] = ll
                row[f'{name}_held_r2'] = r2
                row['score_bins'] = bins
                bootstrap_ll[name] = boot
            ordered = sorted(SHAPES, key=lambda name: row[f'{name}_held_ll'], reverse=True)
            best, runner_up = ordered[:2]
            row.update(best_shape=best, next_shape=runner_up)
            for label, other in [('flat', 'flat'), ('next', runner_up)]:
                difference = interval_difference(row[f'{best}_held_ll'], row[f'{other}_held_ll'],
                                                  bootstrap_ll[best], bootstrap_ll[other])
                row[f'll_gain_{label}'] = difference['gain']
                row[f'll_gain_{label}_lower'] = difference['lower']
                row[f'll_gain_{label}_upper'] = difference['upper']
            row['beats_flat'] = row['ll_gain_flat_lower'] > 0
            row['preferred'] = ('flat' if not row['beats_flat'] else
                                ('undecided' if row['ll_gain_next_lower'] <= 0 <= row['ll_gain_next_upper'] else best))
            row['best_shape_held_r2'] = row[f'{best}_held_r2']
            trial, bins = np.where(np.isfinite(signal))
            y, t, interval = signal[trial, bins].astype(float), times[bins], intervals[trial]
            for name in SHAPES:
                params, ok = fit_field(name, t, interval, y)
                if not ok:
                    cell_failures.append(dict(model=name, stage='full'))
                names = ['a0', 'a', 'mu', 'sigma' if name == 'gaussian' else 'tau']
                for param_name, value in zip(names, params):
                    row[f'{name}_{param_name}'] = value
                lower, upper = BOUNDS[name]
                flags = np.isclose(params[2:], lower, atol=1e-5, rtol=0) | np.isclose(params[2:], upper, atol=1e-5, rtol=0)
                for param_name, flag in zip(names[2:], flags):
                    row[f'{name}_{param_name}_at_bound'] = bool(flag)
                row[f'{name}_a_at_bound'] = bool(params[1] <= 1e-5)
            degree = max(range(1, 5), key=lambda k: row[f'poly{k}_held_r2'])
            row['best_poly_degree'] = degree
            row['best_poly_held_r2'] = row[f'poly{degree}_held_r2']
            if row['preferred'] in SHAPES:
                name = row['preferred']
                params, rho, ok = fit_scaling(name, t, interval, y, intervals.mean())
                if not ok:
                    cell_failures.append(dict(model=name, stage='scaled_full_grid'))
                row.update(rho=rho, rho_at_bound=rho in (-1., 2.))
                for param_name, value in zip(['a0', 'a', 'mu', 'sigma' if name == 'gaussian' else 'tau'], params):
                    row[f'scaled_{param_name}'] = value
                lower, upper = BOUNDS[name]
                flags = np.isclose(params[2:], lower, atol=1e-5, rtol=0) | np.isclose(params[2:], upper, atol=1e-5, rtol=0)
                row.update(scaled_mu_at_bound=bool(flags[0]), scaled_width_at_bound=bool(flags[1]),
                           scaled_a_at_bound=bool(params[1] <= 1e-5))
                held, failed = cv_prediction(name, signal, times, intervals, folds, scaling=True)
                cell_failures.extend(failed)
                saved['scaled'][neuron] = held
                ll, r2, _, boot = score_prediction(signal, held, weights)
                difference = interval_difference(ll, row[f'{name}_held_ll'], boot, bootstrap_ll[name])
                row.update(scaled_held_ll=ll, scaled_held_r2=r2,
                           scaling_ll_gain=difference['gain'], scaling_ll_gain_lower=difference['lower'],
                           scaling_ll_gain_upper=difference['upper'])
            row['fit_success'] = not cell_failures
            failures.extend([dict(session=session, neu_id=neuron + 1, **failure) for failure in cell_failures])
            rows.append(row)
            if len(rows) % 10 == 0:
                pd.DataFrame(rows).to_csv(OUTPUT / f'{session}.csv', index=False)
                saved.flush()
                print(session, 'completed neuron', neuron + 1, flush=True)
    pd.DataFrame(rows).to_csv(OUTPUT / f'{session}.csv', index=False)
    elapsed = time.perf_counter() - started
    (OUTPUT / f'{session}_timing.json').write_text(json.dumps(dict(seconds=elapsed, failures=failures), indent=2))
    print(session, 'complete in seconds', elapsed, flush=True)
    return elapsed, len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not json.loads((OUTPUT / 'dataset_status.json').read_text())['alignment_passed']:
        raise ValueError('alignment gate did not pass')
    started = time.perf_counter()
    seconds, neurons = fit_session(SESSIONS_4S[0])
    total = len(pd.read_csv(OUTPUT / 'labels.csv'))
    estimate = seconds * (total - neurons) / neurons / args.workers / 3600
    print('estimated remaining fitting hours after first session:', estimate, flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fit_session, SESSIONS_4S[1:]))
    pd.concat([pd.read_csv(OUTPUT / f'{session}.csv') for session in SESSIONS_4S], ignore_index=True).to_csv(OUTPUT / 'fits_all.csv', index=False)
    (OUTPUT / 'fit_timing.json').write_text(json.dumps(dict(seconds=time.perf_counter() - started), indent=2))


if __name__ == '__main__':
    main()
