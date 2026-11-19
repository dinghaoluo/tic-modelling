'''
Created on Thu Nov 19 2026

compare the next-lick Gaussian with the rise fit at both onset widths

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
from analysis.wider_next_lick_gaussian import NEW, OUTPUT as NEXT, FAILURE_COLUMNS
from analysis.wider_onset_width import OUTPUT as ONSET, WIDTH
from analysis.wider_window_check import OUTPUT as WIDER, read_wider_inputs

OUTPUT = ROOT / 'outputs' / 'wider_next_lick_onset'
CONTRAST = f'{NEW}_minus_rise'
SCORE_COLUMNS = ['mean_contrast', 'lower', 'upper', 'flat_residual_variance']


#%% recover the held-out predictions
def fit_session(session):
    old = pd.read_csv(NEXT / 'cells.csv').query('session == @session and contrast == @CONTRAST')
    narrow = pd.read_csv(ONSET / 'cells.csv').query(
        "session == @session and group == 'tic_up' and onset_width == @WIDTH")
    old = old.set_index('neu_id', verify_integrity=True)
    narrow = narrow.set_index('neu_id', verify_integrity=True)
    rows, failures = [], []
    with h5py.File(WIDER / f'{session}_wider_oof.h5', 'r') as saved:
        data, folds, cells = read_wider_inputs(session, saved)
        ramp_up = cells[cells.group.eq('tic_up')]
        np.testing.assert_array_equal(ramp_up.neu_id, old.index)
        np.testing.assert_array_equal(ramp_up.neu_id, narrow.index)
        times, intervals, split = data['t'], data['T_k'], folds['folds']
        for i, cell in ramp_up.iterrows():
            signal = data['dff'][i].astype(float)
            next_lick, gaussian_failures = cv_prediction('gaussian', signal, times, intervals, split,
                                                       next_lick=True)
            rise, rise_failures = cv_prediction('rise', signal, times, intervals, split,
                                               onset_width=WIDTH)
            failures.extend(dict(failure, session=session, neu_id=cell.neu_id, model=NEW)
                            for failure in gaussian_failures)
            failures.extend(dict(failure, session=session, neu_id=cell.neu_id) for failure in rise_failures)

            flat = saved['flat'][i]
            resamples = folds['bootstrap_trials']
            # the saved contrasts check both refits against the original held-out comparisons
            for reference, model_a, gaussian, rise_prediction in [
                    (old.loc[cell.neu_id], NEW, next_lick, saved['rise'][i]),
                    (narrow.loc[cell.neu_id], 'gaussian', saved['gaussian'][i], rise)]:
                predictions = {'flat': flat, model_a: gaussian, 'rise': rise_prediction}
                score, _ = scored_contrast(signal, predictions, model_a, 'rise', resamples)
                np.testing.assert_allclose([score[col] for col in SCORE_COLUMNS],
                                           reference[SCORE_COLUMNS].to_numpy(float), rtol=1e-12, atol=1e-15)
                if score['decision'] != reference.decision:
                    raise ValueError(f'{session}, neuron {cell.neu_id}: saved decision differs')

            score, _ = scored_contrast(signal, {'flat': flat, NEW: next_lick, 'rise': rise},
                                       NEW, 'rise', resamples)
            rows.append(dict(session=session, neu_id=cell.neu_id, group=cell.group,
                             onset_width=WIDTH, contrast=CONTRAST, model_a=NEW, model_b='rise', **score))
    print(session, len(rows), 'cells;', len(failures), 'fit failures', flush=True)
    return rows, failures


#%% summary
def main():
    workers = min(6, os.cpu_count() or 1, len(SESSIONS_4S))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(fit_session, SESSIONS_4S))
    new = pd.DataFrame([row for rows, _ in results for row in rows])
    failures = pd.DataFrame([row for _, rows in results for row in rows], columns=FAILURE_COLUMNS)
    old = pd.read_csv(NEXT / 'cells.csv').query('contrast == @CONTRAST').assign(onset_width=.2)
    summary = pd.DataFrame([dict(onset_width=width, contrast=CONTRAST, **summary_row(cells))
                            for width, cells in [(.2, old), (WIDTH, new)]])
    np.testing.assert_array_equal(summary.n_sessions, len(SESSIONS_4S))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.concat([old, new], ignore_index=True).to_csv(OUTPUT / 'cells.csv', index=False)
    summary.to_csv(OUTPUT / 'summary.csv', index=False)
    failures.to_csv(OUTPUT / 'failures.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
