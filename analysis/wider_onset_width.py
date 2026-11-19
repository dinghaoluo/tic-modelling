'''
Created on Wed Nov 18 2026

refit the wider-window exponentials with a 0.1 s Gaussian onset

@author: Dinghao Luo
'''

#%% imports
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rec_list import SESSIONS_4S
from utils.rf_models import fit_field
from analysis.compare_models_trial import summary_row
from analysis.fit_shapes import cv_prediction
from analysis.recovery_check import GROUPS, matched_shape, scored_contrast
from analysis.wider_window_check import OUTPUT as WIDER, read_wider_inputs

OUTPUT = ROOT / 'outputs' / 'wider_onset_width'
WIDTH = .1
MODELS = ['decay', 'rise']
FAILURE_COLUMNS = ['onset_width', 'session', 'neu_id', 'group', 'model', 'fold', 'stage']


#%% fit and score a session
def fit_session(session):
    rows, parameter_rows, failures = [], [], []
    with h5py.File(WIDER / f'{session}_wider_oof.h5', 'r') as saved:
        data, folds, cells = read_wider_inputs(session, saved)
        times, intervals = data['t'], data['T_k']
        for i, cell in enumerate(cells.itertuples(index=False)):
            key = dict(session=session, neu_id=cell.neu_id, group=cell.group)
            signal = data['dff'][i].astype(float)
            trial, bins = np.where(np.isfinite(signal))
            predictions = {'flat': saved['flat'][i], 'gaussian': saved['gaussian'][i]}
            full = {}
            for model in MODELS:
                predictions[model], failed = cv_prediction(model, signal, times, intervals,
                                                            folds['folds'], onset_width=WIDTH)
                params, success = fit_field(model, times[bins], intervals[trial], signal[trial, bins],
                                            onset_width=WIDTH)
                if not success:
                    failed.append(dict(model=model, stage='full'))
                failures.extend(dict(**key, onset_width=WIDTH, **failure) for failure in failed)
                full.update({f'{model}_{name}': value for name, value in
                             zip(['a0', 'a', 'mu', 'tau'], params)})
            model = matched_shape(cell.group)
            score, _ = scored_contrast(signal, predictions, 'gaussian', model, folds['bootstrap_trials'])
            rows.append(dict(**key, onset_width=WIDTH, model_a='gaussian', model_b=model,
                             contrast=f'gaussian_minus_{model}', primary=True, **score))
            parameter_rows.append(dict(**key, onset_width=WIDTH, **full))
    print(session, len(rows), 'cells;', len(failures), 'fit failures', flush=True)
    return rows, parameter_rows, failures


#%% compare with 0.2 s
def main():
    workers = min(6, os.cpu_count() or 1, len(SESSIONS_4S))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(fit_session, SESSIONS_4S))
    new = pd.DataFrame([row for rows, _, _ in results for row in rows])
    params = pd.DataFrame([row for _, rows, _ in results for row in rows])
    failures = pd.DataFrame([row for _, _, rows in results for row in rows], columns=FAILURE_COLUMNS)

    old = pd.concat([pd.read_csv(WIDER / f'{session}_wider_contrasts.csv')
                     for session in SESSIONS_4S], ignore_index=True)
    old = old.loc[old.primary].assign(onset_width=.2)
    old_params = pd.concat([pd.read_csv(WIDER / f'{session}_wider_parameters.csv')
                            for session in SESSIONS_4S], ignore_index=True)
    keys = ['session', 'neu_id', 'group']
    paired = old.merge(new, on=keys + ['contrast'], how='outer', validate='one_to_one',
                       suffixes=('_old', '_new'), indicator=True)
    if not paired['_merge'].eq('both').all():
        raise ValueError('saved and refitted primary contrasts cover different cells')
    np.testing.assert_allclose(paired.flat_residual_variance_old, paired.flat_residual_variance_new,
                               rtol=1e-12, atol=0.)
    old_failures = pd.concat([pd.read_csv(WIDER / f'{session}_wider_failures.csv')
                              for session in SESSIONS_4S], ignore_index=True)
    old_failures = old_failures.merge(old_params[keys], on=keys[:2], validate='many_to_one').assign(onset_width=.2)

    summary = []
    for group in GROUPS:
        for width, cells, fitted in [(.2, old, old_params), (WIDTH, new, params)]:
            part = cells[cells.group.eq(group)]
            group_params = fitted[fitted.group.eq(group)]
            summary.append(dict(group=group, onset_width=width, contrast=part.contrast.iloc[0],
                                **summary_row(part), median_decay_tau=group_params.decay_tau.median(),
                                median_rise_tau=group_params.rise_tau.median()))
    summary = pd.DataFrame(summary)
    np.testing.assert_array_equal(summary.n_sessions, len(SESSIONS_4S))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.concat([old, new], ignore_index=True).to_csv(OUTPUT / 'cells.csv', index=False)
    params.to_csv(OUTPUT / 'parameters.csv', index=False)
    summary.to_csv(OUTPUT / 'summary.csv', index=False)
    pd.concat([old_failures, failures], ignore_index=True)[FAILURE_COLUMNS].to_csv(OUTPUT / 'failures.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
