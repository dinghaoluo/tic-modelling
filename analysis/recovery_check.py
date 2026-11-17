# -*- coding: utf-8 -*-
'''
Created on Thu Nov  5 18:24:50 2026

recover known field shapes with trial-wise sign-flipped held-out residuals

@author: Dinghao Luo
'''

#%% imports
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rec_list import SESSIONS_4S
from utils.rf_models_v1_2 import predict
from utils.trial_error_scores import trial_mse
from analysis.fit_shapes_v1_2 import cv_prediction
from analysis.compare_models_trial import trial_contrast, trial_loglikelihood

#%% constants
OUTPUT = ROOT / 'outputs' / 'undecided_check'
FIGURES = ROOT / 'figures' / 'undecided_check'
FITS = ROOT / 'outputs' / 'fits_v1_2'
GROUPS = ['tic_down', 'tic_up', 'time_cell']
LABELS = ['Ramp-down TIC', 'Ramp-up TIC', 'Time cell']
SEED = 20261105

#%% synthetic trials
def sign_flip_signal(truth, real_signal, held_prediction, signs):
    '''one sign per trial preserves the complete residual time course and its missing bins'''
    if truth.shape != real_signal.shape or held_prediction.shape != real_signal.shape:
        raise ValueError('synthetic signal arrays differ in shape')
    signs = np.asarray(signs)
    if signs.shape != (len(real_signal),) or not np.isin(signs, [-1, 1]).all():
        raise ValueError('expected one positive or negative sign per trial')
    observed = np.isfinite(real_signal)
    if not np.isfinite(held_prediction[observed]).all() or not np.isfinite(truth[observed]).all():
        raise ValueError('truth or residual prediction is missing observed bins')
    return np.where(observed, truth + signs[:, None] * (real_signal - held_prediction), np.nan)


def matched_shape(group):
    return 'rise' if group == 'tic_up' else 'decay'


def fitted_params(cell, shape):
    return cell[[f'{shape}_a0', f'{shape}_a', f'{shape}_mu',
                 f'{shape}_sigma' if shape == 'gaussian' else f'{shape}_tau']].to_numpy(float)


def scored_contrast(signal, predictions, a, b, resamples):
    residual = (signal - predictions['flat'])[np.isfinite(signal)]
    variance = np.var(residual, ddof=0)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError('flat held-out residual variance is not positive')
    errors = {name: trial_mse(signal, prediction) for name, prediction in predictions.items()}
    difference = trial_contrast(errors[a], errors[b], variance)
    lower, upper = np.percentile(difference[resamples].mean(axis=1), [2.5, 97.5])
    return dict(mean_contrast=float(difference.mean()), lower=float(lower), upper=float(upper),
                decision=a if lower > 0 else b if upper < 0 else 'undecided',
                flat_residual_variance=float(variance)), errors


def sample_cells():
    fits = pd.read_csv(FITS / 'fits_all.csv')
    reliability = pd.read_csv(ROOT / 'outputs' / 'label_stability' / 'cells.csv',
                              usecols=['session', 'neu_id', 'split_half_r'])
    fits = fits.merge(reliability, on=['session', 'neu_id'], validate='one_to_one', how='left')
    if fits.split_half_r.isna().any():
        raise ValueError('missing saved split-half reliability')
    rng = np.random.default_rng(SEED)
    samples, allocations = [], []
    for group in GROUPS:
        cells = fits[fits.group == group]
        counts = np.array([(cells.session == session).sum() for session in SESSIONS_4S])
        quotas = 150 * counts / counts.sum()
        sizes = np.floor(quotas).astype(int)
        remainder = 150 - sizes.sum()
        sizes[np.argsort(-(quotas - sizes), kind='stable')[:remainder]] += 1
        for session, size, count in zip(SESSIONS_4S, sizes, counts):
            part = cells[cells.session == session]
            if size > len(part):
                raise ValueError('proportional sample exceeds available cells')
            chosen = part.iloc[rng.choice(len(part), int(size), replace=False)].copy()
            chosen['synthetic_seed'] = rng.integers(0, 2 ** 32, size=len(chosen), dtype=np.uint64)
            samples.append(chosen)
            allocations.append(dict(group=group, session=session, available=int(count), sampled=int(size)))
    pd.concat(samples, ignore_index=True).to_csv(OUTPUT / 'recovery_sample.csv', index=False)
    pd.DataFrame(allocations).to_csv(OUTPUT / 'recovery_allocation.csv', index=False)


#%% session recovery
def recovery_session(session):
    started = time.perf_counter()
    sample = pd.read_csv(OUTPUT / 'recovery_sample.csv')
    sample = sample[sample.session == session]
    data = np.load(ROOT / 'data' / 'binned_v1_1' / f'{session}.npz')
    dff, times, intervals = data['dff'], data['t'], data['T_k']
    folds = np.load(FITS / f'{session}_folds.npz')
    np.testing.assert_array_equal(folds['trial_rows'], data['trial_rows'])
    trial, bins = np.where(np.isfinite(dff[0]))
    rows, failures, signs_saved = [], [], {}
    with h5py.File(FITS / f'{session}_oof.h5', 'r') as saved:
        np.testing.assert_array_equal(saved['trial_rows'][:], data['trial_rows'] + 1)
        for _, cell in sample.iterrows():
            neuron = int(cell.neu_id) - 1
            signal = dff[neuron].astype(float)
            exponential = matched_shape(cell.group)
            rng = np.random.default_rng(int(cell.synthetic_seed))
            for truth_shape in ['gaussian', exponential]:
                params = fitted_params(cell, truth_shape)
                truth = predict(truth_shape, np.broadcast_to(times, signal.shape), intervals[:, None], params)
                signs = rng.choice([-1, 1], size=len(intervals))
                signs_saved[f'{neuron + 1}_{truth_shape}'] = signs
                synthetic = sign_flip_signal(truth, signal, saved[truth_shape][neuron], signs)
                predictions, cell_failures = {}, []
                for name in ['flat', 'gaussian', exponential]:
                    predictions[name], failed = cv_prediction(name, synthetic, times, intervals, folds['folds'])
                    cell_failures.extend(failed)
                result, _ = scored_contrast(synthetic, predictions, 'gaussian', exponential, folds['bootstrap_trials'])
                decision = result['decision']
                outcome = ('undecided' if decision == 'undecided' else 'correct' if decision == truth_shape else 'wrong')
                rows.append(dict(session=session, neu_id=neuron + 1, group=cell.group, truth_shape=truth_shape,
                                 matched_exponential=exponential, split_half_r=cell.split_half_r,
                                 synthetic_seed=int(cell.synthetic_seed), outcome=outcome,
                                 fit_success=not cell_failures, **result))
                failures.extend([dict(neu_id=neuron + 1, truth_shape=truth_shape, **failure) for failure in cell_failures])
            if len(rows) % 20 == 0:
                print(session, 'recovery datasets', len(rows), flush=True)
    pd.DataFrame(rows).to_csv(OUTPUT / f'{session}_recovery.csv', index=False)
    np.savez(OUTPUT / f'{session}_recovery_signs.npz', **signs_saved)
    status = dict(session=session, seconds=time.perf_counter() - started, failures=failures)
    (OUTPUT / f'{session}_recovery_status.json').write_text(json.dumps(status, indent=2))
    print(session, 'recovery complete', status['seconds'], flush=True)
    return status


def recovery_summary():
    frame = pd.concat([pd.read_csv(OUTPUT / f'{session}_recovery.csv') for session in SESSIONS_4S], ignore_index=True)
    rows = []
    for subset, selected in [('all_sampled', frame), ('split_half_r_above_0.8', frame[frame.split_half_r > .8])]:
        for group in GROUPS:
            for truth in ['gaussian', matched_shape(group)]:
                cells = selected[(selected.group == group) & (selected.truth_shape == truth)]
                for outcome in ['correct', 'wrong', 'undecided']:
                    count, n = int((cells.outcome == outcome).sum()), len(cells)
                    interval = binomtest(count, n).proportion_ci(confidence_level=.95, method='exact') if n else None
                    rows.append(dict(subset=subset, group=group, truth_shape=truth, outcome=outcome,
                                     n_cells=n, count=count, percentage=100 * count / n if n else np.nan,
                                     lower=100 * interval.low if interval else np.nan,
                                     upper=100 * interval.high if interval else np.nan,
                                     interval_method='Clopper-Pearson exact'))
    frame.to_csv(OUTPUT / 'recovery_cells.csv', index=False)
    pd.DataFrame(rows).to_csv(OUTPUT / 'recovery_summary.csv', index=False)


#%% main
def main():
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sample_cells()
    workers = min(12, os.cpu_count() or 1, len(SESSIONS_4S))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(recovery_session, SESSIONS_4S))
    recovery_summary()
    (OUTPUT / 'recovery_timing.json').write_text(json.dumps(dict(seconds=time.perf_counter() - started, workers=workers), indent=2))


if __name__ == '__main__':
    main()
