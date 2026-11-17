# -*- coding: utf-8 -*-
'''
Created on Wed Nov  4 23:46:13 2026

recompute Kori's adjacent-trial validity rule from cue-aligned lick indices

@author: Dinghao Luo
'''

#%% imports
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rec_list import SESSIONS_4S, session_path
from utils.matlab_inputs import load_numeric

#%% trial selection
def cue_trial_fields(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'missing alignCue source: {path}')
    if not h5py.is_hdf5(path):
        data = loadmat(str(path), variable_names=['trialsCue'], simplify_cells=True)
        if 'trialsCue' not in data:
            raise ValueError(f'missing trialsCue: {path}')
        trial = data['trialsCue']
        return (np.asarray(trial['startLfpInd']).ravel(),
                np.asarray(trial['endLfpInd']).ravel(),
                np.asarray(trial['lickLfpInd'], dtype=object).ravel())
    with h5py.File(path, 'r') as saved:
        if 'trialsCue' not in saved:
            raise ValueError(f'missing trialsCue: {path}')
        trial = saved['trialsCue']
        start = np.asarray(trial['startLfpInd']).ravel()
        end = np.asarray(trial['endLfpInd']).ravel()
        refs = np.asarray(trial['lickLfpInd']).ravel()
        licks = []
        for ref in refs:
            cell = saved[ref]
            # MATLAB stores an empty cell's dimensions rather than lick indices
            licks.append(np.array([]) if cell.attrs.get('MATLAB_empty', 0)
                         else np.asarray(cell).ravel())
    return start, end, licks


def valid_trial_rows(start, end, licks):
    '''return sorted 1-based dFFGF rows whose two adjacent MATLAB trials pass'''
    if len(start) != len(end) or len(start) != len(licks):
        raise ValueError('trialsCue field lengths differ')
    end_from_last = np.zeros(len(start))
    start_from_first = np.zeros(len(start))
    for trial in range(1, len(start) - 1):
        lick = np.asarray(licks[trial]).ravel()
        first, last = (lick[0], lick[-1]) if lick.size else (0., 0.)
        end_from_last[trial] = (end[trial] - last) / 500
        start_from_first[trial] = (first - start[trial]) / 500
    passes = (end_from_last > .9) & (start_from_first > 1.)
    valid = np.flatnonzero(passes[:-1] & passes[1:]) + 1
    return valid.astype(np.int64), end_from_last, start_from_first


#%% main
def main():
    target = ROOT / 'data' / 'valid_trials'
    output = ROOT / 'outputs' / 'initial_fits'
    target.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    checks = []
    for session in SESSIONS_4S:
        folder = session_path(session)
        source = folder / f'{session}_DataStructure_mazeSection1_TrialType1_alignCue_msess1.mat'
        try:
            start, end, licks = cue_trial_fields(source)
        except (FileNotFoundError, KeyError, ValueError) as error:
            checks.append(dict(session=session, source=str(source), error=str(error)))
            raise
        valid, end_from_last, start_from_first = valid_trial_rows(start, end, licks)
        np.save(target / f'{session}.npy', valid)
        np.savez(output / f'{session}_trial_rule.npz', EndfromLL=end_from_last,
                 StartfromL=start_from_first)
        check = dict(session=session, trials=len(start), valid_trials=len(valid),
                     compared=False, source=str(source))
        reference = folder / 'ValidTr_FirstLastLick2.mat'
        if reference.is_file():
            expected = load_numeric(reference)['ValidTrAll'].ravel()
            check.update(compared=True, exact_match=bool(np.array_equal(valid, expected)),
                         computed_only=np.setdiff1d(valid, expected).astype(int).tolist(),
                         source_only=np.setdiff1d(expected, valid).astype(int).tolist())
        checks.append(check)
        print(check, flush=True)
    matched = all(check.get('exact_match', True) for check in checks)
    if not matched:
        raise ValueError('computed valid trials differ from ValidTrAll; stop before dataset construction')


if __name__ == '__main__':
    main()
