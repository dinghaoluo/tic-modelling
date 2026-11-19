'''
Created on Wed Nov 18 2026

compare next-lick and last-lick Gaussian fits to ramp-up TICs

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
from analysis.compare_models_trial import summary_row
from analysis.fit_shapes import cv_prediction
from analysis.recovery_check import scored_contrast
from analysis.wider_window_check import OUTPUT as WIDER, read_wider_inputs

OUTPUT = ROOT / 'outputs' / 'wider_next_lick_gaussian'
NEW = 'gaussian_next_lick'
PAIRS = [(NEW, 'rise'), (NEW, 'gaussian'), ('gaussian', 'rise')]
FAILURE_COLUMNS = ['session', 'neu_id', 'model', 'fold', 'stage']


#%% fit and score a session
def fit_session(session):
    cells, failures = [], []
    with h5py.File(WIDER / f'{session}_wider_oof.h5', 'r') as saved:
        data, folds, params = read_wider_inputs(session, saved)
        baseline = pd.read_csv(WIDER / f'{session}_wider_contrasts.csv')
        baseline = baseline.loc[baseline.group.eq('tic_up') & baseline.contrast.eq('gaussian_minus_rise')]
        baseline = baseline.set_index('neu_id', verify_integrity=True)
        ramp_up = params[params.group.eq('tic_up')]
        np.testing.assert_array_equal(ramp_up.neu_id, baseline.index)
        for i, cell in ramp_up.iterrows():
            signal = data['dff'][i].astype(float)
            next_lick, failed = cv_prediction('gaussian', signal, data['t'], data['T_k'],
                                              folds['folds'], next_lick=True)
            key = dict(session=session, neu_id=cell.neu_id, group=cell.group)
            failures.extend(dict(session=session, neu_id=cell.neu_id, model=NEW,
                                 fold=failure['fold'], stage=failure['stage']) for failure in failed)
            predictions = {name: saved[name][i] for name in ['flat', 'gaussian', 'rise']}
            predictions[NEW] = next_lick
            for model_a, model_b in PAIRS:
                contrast = f'{model_a}_minus_{model_b}'
                use = {name: predictions[name] for name in ['flat', model_a, model_b]}
                score, _ = scored_contrast(signal, use, model_a, model_b, folds['bootstrap_trials'])
                if model_a == 'gaussian':
                    old = baseline.loc[cell.neu_id]
                    columns = ['mean_contrast', 'lower', 'upper', 'flat_residual_variance']
                    np.testing.assert_allclose([score[col] for col in columns], old[columns].to_numpy(float),
                                               rtol=1e-12, atol=1e-15)
                    if score['decision'] != old.decision:
                        raise ValueError(f'{session}, neuron {cell.neu_id}: saved decision differs')
                cells.append(dict(**key, contrast=contrast, model_a=model_a, model_b=model_b, **score))
    old_failures = pd.read_csv(WIDER / f'{session}_wider_failures.csv')
    old_failures = old_failures[old_failures.neu_id.isin(ramp_up.neu_id) &
                                old_failures.model.isin(['gaussian', 'rise', 'flat']) &
                                old_failures.stage.eq('cv')]
    print(session, len(ramp_up), 'cells;', len(failures), 'next-lick fit failures', flush=True)
    return cells, failures + old_failures.to_dict('records')


#%% summary
def main():
    workers = min(6, os.cpu_count() or 1, len(SESSIONS_4S))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(fit_session, SESSIONS_4S))
    cells = pd.DataFrame([row for rows, _ in results for row in rows])
    failures = pd.DataFrame([row for _, rows in results for row in rows], columns=FAILURE_COLUMNS)
    summary = pd.DataFrame([
        dict(contrast=f'{a}_minus_{b}', model_a=a, model_b=b, **summary_row(
             cells[cells.model_a.eq(a) & cells.model_b.eq(b)]))
        for a, b in PAIRS
    ])
    np.testing.assert_array_equal(summary.n_sessions, len(SESSIONS_4S))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    cells.to_csv(OUTPUT / 'cells.csv', index=False)
    summary.to_csv(OUTPUT / 'summary.csv', index=False)
    failures.to_csv(OUTPUT / 'failures.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
