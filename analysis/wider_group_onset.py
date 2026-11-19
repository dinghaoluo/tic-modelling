'''
Created on Thu Nov 19 2026

compare the Gaussian minus decay group differences at both onset widths

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
from analysis.fit_shapes import cv_prediction
from analysis.recovery_check import scored_contrast
from analysis.wider_group_difference import OUTPUT as GROUP_DIFFERENCE, group_tables
from analysis.wider_onset_width import OUTPUT as ONSET, WIDTH, FAILURE_COLUMNS
from analysis.wider_window_check import OUTPUT as WIDER, read_wider_inputs

OUTPUT = ROOT / 'outputs' / 'wider_group_onset'
CONTRAST = 'gaussian_minus_decay'


#%% recover the missing ramp-up contrasts
def fit_session(session):
    rows, failures = [], []
    with h5py.File(WIDER / f'{session}_wider_oof.h5', 'r') as saved:
        data, folds, cells = read_wider_inputs(session, saved)
        ramp_up = cells[cells.group.eq('tic_up')]
        for i, cell in ramp_up.iterrows():
            key = dict(session=session, neu_id=cell.neu_id, group=cell.group, onset_width=WIDTH)
            signal = data['dff'][i].astype(float)
            decay, failed = cv_prediction('decay', signal, data['t'], data['T_k'],
                                          folds['folds'], onset_width=WIDTH)
            failures.extend(dict(**key, **failure) for failure in failed)
            predictions = {'flat': saved['flat'][i], 'gaussian': saved['gaussian'][i], 'decay': decay}
            score, _ = scored_contrast(signal, predictions, 'gaussian', 'decay', folds['bootstrap_trials'])
            rows.append(dict(**key, contrast=CONTRAST, model_a='gaussian', model_b='decay',
                             primary=False, **score))
    print(session, len(rows), 'cells;', len(failures), 'fit failures', flush=True)
    return rows, failures


#%% group differences
def main():
    workers = min(6, os.cpu_count() or 1, len(SESSIONS_4S))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(fit_session, SESSIONS_4S))
    ramp_up = pd.DataFrame([row for rows, _ in results for row in rows])
    failures = pd.DataFrame([row for _, rows in results for row in rows], columns=FAILURE_COLUMNS)

    saved = pd.read_csv(ONSET / 'cells.csv')
    new = pd.concat([saved.query('onset_width == @WIDTH and contrast == @CONTRAST'), ramp_up],
                    ignore_index=True)
    old = pd.concat([pd.read_csv(WIDER / f'{session}_wider_contrasts.csv')
                     for session in SESSIONS_4S], ignore_index=True)
    old = old.loc[old.contrast.eq(CONTRAST)].assign(onset_width=.2)
    keys = ['session', 'neu_id', 'group']
    paired = old.merge(new, on=keys, how='outer', validate='one_to_one',
                       suffixes=('_old', '_new'), indicator=True)
    if not paired['_merge'].eq('both').all():
        raise ValueError('the two onset widths cover different cells')
    np.testing.assert_allclose(paired.flat_residual_variance_old, paired.flat_residual_variance_new,
                               rtol=1e-12, atol=0.)

    saved_failures = pd.read_csv(ONSET / 'failures.csv').query(
        "stage == 'cv' and model in ['gaussian', 'decay']")
    expected = saved_failures.query("onset_width == @WIDTH and group == 'tic_up'")
    columns = ['session', 'neu_id', 'fold']
    pd.testing.assert_frame_equal(failures[columns].reset_index(drop=True),
                                  expected[columns].reset_index(drop=True), check_dtype=False)

    old_medians = pd.read_csv(GROUP_DIFFERENCE / 'session_medians.csv', index_col='session')
    old_tests = pd.read_csv(GROUP_DIFFERENCE / 'group_tests.csv')
    medians, tests = group_tables(new)
    both_medians = pd.concat({.2: old_medians, WIDTH: medians}, names=['onset_width', 'session'])
    both_tests = pd.concat([old_tests.assign(onset_width=.2), tests.assign(onset_width=WIDTH)],
                           ignore_index=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.concat([old, new], ignore_index=True).to_csv(OUTPUT / 'cells.csv', index=False)
    both_medians.to_csv(OUTPUT / 'session_medians.csv')
    both_tests.to_csv(OUTPUT / 'group_tests.csv', index=False)
    saved_failures.to_csv(OUTPUT / 'failures.csv', index=False)
    print(both_medians.to_string())
    print(both_tests.to_string(index=False))


if __name__ == '__main__':
    main()
