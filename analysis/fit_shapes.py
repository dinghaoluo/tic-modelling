'''
Created on Thu Nov  5 11:24:40 2026
Updated on Wed Nov 18 2026

fit single-trial fields and compare paired out-of-fold trial errors

@author: Dinghao Luo
'''

#%% imports
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rec_list import SESSIONS_4S
from utils.rf_models import BOUNDS, fit_field, predict, fit_scaling
from utils.trial_error_scores import trial_mse, paired_gain, alignment_scores

#%% constants
OUTPUT = ROOT / 'outputs' / 'fits'
DATA = ROOT / 'data' / 'binned'
SHAPES = ['gaussian', 'decay', 'rise']
MODELS = ['flat', *SHAPES, 'poly1', 'poly2', 'poly3', 'poly4']
SEED = 20261104

#%% cross-validation
def cv_prediction(name, signal, times, intervals, folds, scaling=False, *,
                  onset_width=.2, next_lick=False):
    '''fit observed single-trial bins, never a trial-averaged training trace'''
    trial, bins = np.where(np.isfinite(signal))
    observed_values, bin_times, interval = signal[trial, bins].astype(float), times[bins], intervals[trial]
    if next_lick:
        if name != 'gaussian':
            raise ValueError('the next-lick alignment option is for Gaussian fits')
        bin_times = interval - bin_times
    held = np.full(signal.shape, np.nan)
    failures = []
    for fold in range(5):
        train = folds[trial] != fold
        test = ~train
        if name.startswith('poly'):
            degree = int(name[-1])
            design = np.polynomial.polynomial.polyvander(bin_times / 7, degree)
            params = np.linalg.lstsq(design[train], observed_values[train], rcond=None)[0]
            prediction = design[test] @ params
            success = True
        else:
            rho = 0.
            if scaling:
                params, rho, success = fit_scaling(name, bin_times[train], interval[train], observed_values[train],
                                                     intervals.mean(), onset_width=onset_width)
            else:
                params, success = fit_field(name, bin_times[train], interval[train], observed_values[train],
                                            onset_width=onset_width)
            prediction = predict(name, bin_times[test], interval[test], params, rho, intervals.mean(),
                                 onset_width=onset_width)
        held[trial[test], bins[test]] = prediction
        if not success:
            failures.append(dict(model=name, fold=fold, stage='scaled_cv' if scaling else 'cv'))
    return held, failures


#%% session fits
def fit_session(session):
    data = np.load(DATA / f'{session}.npz')
    dff, intervals, times = data['dff'], data['T_k'], data['t']
    labels = pd.read_csv(ROOT / 'outputs' / 'dataset' / 'labels.csv').query('session == @session').reset_index(drop=True)
    n_trials = len(intervals)
    if n_trials < 10:
        raise ValueError(f'{session}: fewer than ten kept trials')
    rng = np.random.default_rng(SEED + SESSIONS_4S.index(session))
    folds = np.empty(n_trials, dtype=int)
    for fold, indices in enumerate(np.array_split(rng.permutation(n_trials), 5)):
        folds[indices] = fold
    resamples = rng.integers(0, n_trials, size=(1000, n_trials))
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
            errors = {}
            cell_failures = []
            for name in MODELS:
                held, failed = cv_prediction(name, signal, times, intervals, folds)
                saved[name][neuron] = held
                cell_failures.extend(failed)
                scores = alignment_scores(signal, held, times, intervals)
                row.update({f'{name}_{key}': value for key, value in scores.items()})
                errors[name] = trial_mse(signal, held)
                row[f'{name}_held_mse'] = float(errors[name].mean())
            ordered = sorted(SHAPES, key=lambda name: row[f'{name}_held_mse'])
            best, runner_up = ordered[:2]
            row.update(best_shape=best, next_shape=runner_up)
            for label, other in [('flat', 'flat'), ('next', runner_up)]:
                difference = paired_gain(errors[best], errors[other], resamples)
                for key, value in difference.items():
                    row[f'mse_gain_{label}' + ('' if key == 'gain' else f'_{key}')] = value
            row['beats_flat'] = row['mse_gain_flat_lower'] > 0
            row['preferred'] = ('flat' if not row['beats_flat'] else
                                ('undecided' if row['mse_gain_next_lower'] <= 0 <= row['mse_gain_next_upper'] else best))
            row['best_shape_held_r2'] = row[f'{best}_held_r2_last']
            row['best_shape_held_r2_first'] = row[f'{best}_held_r2_first']
            trial, bins = np.where(np.isfinite(signal))
            observed_values, bin_times, interval = signal[trial, bins].astype(float), times[bins], intervals[trial]
            for name in SHAPES:
                params, success = fit_field(name, bin_times, interval, observed_values)
                if not success:
                    cell_failures.append(dict(model=name, stage='full'))
                names = ['a0', 'a', 'mu', 'sigma' if name == 'gaussian' else 'tau']
                for param_name, value in zip(names, params):
                    row[f'{name}_{param_name}'] = value
                lower, upper = BOUNDS[name]
                flags = np.isclose(params[2:], lower, atol=1e-5, rtol=0) | np.isclose(params[2:], upper, atol=1e-5, rtol=0)
                for param_name, flag in zip(names[2:], flags):
                    row[f'{name}_{param_name}_at_bound'] = bool(flag)
                row[f'{name}_a_at_bound'] = bool(params[1] <= 1e-5)
            row['gaussian_class'] = ('field' if row['gaussian_mu'] - row['gaussian_sigma'] > 0
                                     and row['gaussian_mu'] + row['gaussian_sigma'] < np.median(intervals)
                                     else 'monotonic')
            row['session_median_T_k'] = float(np.median(intervals))
            degree = max(range(1, 5), key=lambda degree: row[f'poly{degree}_held_r2'])
            row['best_poly_degree'] = degree
            row['best_poly_held_r2'] = row[f'poly{degree}_held_r2']
            if row['beats_flat']:
                name = best
                row['scaled_shape'] = best
                params, rho, success = fit_scaling(name, bin_times, interval, observed_values, intervals.mean())
                if not success:
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
                scores = alignment_scores(signal, held, times, intervals)
                row.update({f'scaled_{key}': value for key, value in scores.items()})
                scaled_error = trial_mse(signal, held)
                row['scaled_held_mse'] = float(scaled_error.mean())
                difference = paired_gain(scaled_error, errors[name], resamples)
                row.update(scaling_mse_gain=difference['gain'], scaling_mse_gain_lower=difference['lower'],
                           scaling_mse_gain_upper=difference['upper'])
            row['fit_success'] = not cell_failures
            failures.extend([dict(session=session, neu_id=neuron + 1, **failure) for failure in cell_failures])
            rows.append(row)
            if len(rows) % 10 == 0:
                pd.DataFrame(rows).to_csv(OUTPUT / f'{session}.csv', index=False)
                saved.flush()
                print(session, 'completed neuron', neuron + 1, flush=True)
    pd.DataFrame(rows).to_csv(OUTPUT / f'{session}.csv', index=False)
    pd.DataFrame([dict(failure, session=session) for failure in failures], columns=['session', 'neu_id', 'model', 'fold', 'stage']).to_csv(OUTPUT / f'{session}_failures.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    alignment = pd.read_csv(ROOT / 'outputs' / 'dataset' / 'alignment_checks.csv')
    if alignment.offset_zero_largest.sum() <= len(alignment) / 2:
        raise ValueError('offset zero does not give the largest drop in most sessions')
    fit_session(SESSIONS_4S[0])
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fit_session, SESSIONS_4S[1:]))
    pd.concat([pd.read_csv(OUTPUT / f'{session}.csv') for session in SESSIONS_4S], ignore_index=True).to_csv(OUTPUT / 'fits_all.csv', index=False)


if __name__ == '__main__':
    main()
