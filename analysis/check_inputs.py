# -*- coding: utf-8 -*-
'''
Created on 4 November 2026

check that every session has its aligned arrays and report their shapes

Reads dFFGF_befLastLick.mat, FirstLick_LL.mat, and the time-cell index of each
session, then writes one row per session to outputs/check_inputs.csv. Nothing
is written to the raw-data folders.

@author: Dinghao Luo
'''

#%% imports
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'utils'))
from project_paths import OUTPUT_ROOT
from rec_list import SESSIONS_4S, session_path

#%% constants
DFF_FILE   = 'dFFGF_befLastLick.mat'
LICK_FILE  = 'FirstLick_LL.mat'
FIELD_FILE = Path('FieldDetectionShuffle') / 'FieldIndexShuffleLastLick99.mat'

COLUMNS = (
    'session', 'mat_format', 'n_neurons', 'n_trials', 'n_bins', 'n_first_lick',
    'n_valid_first_lick', 'first_lick_median_s', 'n_time_cells', 'missing',
    )

#%% helpers
def _is_hdf5(path: Path) -> bool:
    return h5py.is_hdf5(path)

def _dff_shape(path: Path) -> tuple[str, int, int, int]:
    '''return format and neuron, trial, and bin counts of dFFGFArray'''
    if _is_hdf5(path):
        with h5py.File(path, 'r') as f:
            cells = f['dFFGFArray'][()]
            first = f[cells.flat[0]]
            # MATLAB stores arrays column-first, so HDF5 reads time before trials
            n_bins, n_trials = first.shape
            return 'v7.3', int(cells.size), int(n_trials), int(n_bins)

    cells = loadmat(path, variable_names=['dFFGFArray'])['dFFGFArray']
    n_trials, n_bins = cells.flat[0].shape
    return 'v5', int(cells.size), int(n_trials), int(n_bins)

def _load_vector(path: Path, name: str) -> np.ndarray:
    '''read one numeric variable from a small .mat file as a flat array'''
    if _is_hdf5(path):
        with h5py.File(path, 'r') as f:
            return np.asarray(f[name], dtype=float).ravel()
    return np.asarray(loadmat(path, variable_names=[name])[name], dtype=float).ravel()

def _check_session(session: str) -> dict:
    '''report shapes for one session, recording anything missing'''
    root = session_path(session)
    row = {column: '' for column in COLUMNS}
    row['session'] = session
    missing = []

    if (root / DFF_FILE).exists():
        fmt, n_neurons, n_trials, n_bins = _dff_shape(root / DFF_FILE)
        row.update(mat_format=fmt, n_neurons=n_neurons, n_trials=n_trials, n_bins=n_bins)
    else:
        missing.append(DFF_FILE)

    if (root / LICK_FILE).exists():
        first_lick = _load_vector(root / LICK_FILE, 'FirstLick')
        valid = first_lick[first_lick > 0]  # zero marks a trial without a lick
        row.update(
            n_first_lick=first_lick.size,
            n_valid_first_lick=valid.size,
            first_lick_median_s=round(float(np.median(valid)), 3) if valid.size else '',
            )
    else:
        missing.append(LICK_FILE)

    if (root / FIELD_FILE).exists():
        row['n_time_cells'] = _load_vector(root / FIELD_FILE, 'FieldID').size
    else:
        missing.append(str(FIELD_FILE))

    row['missing'] = ';'.join(missing)
    return row

#%% main
def main(out_file: Path) -> None:
    rows = [_check_session(session) for session in tqdm(SESSIONS_4S, desc='sessions')]

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)
    print(f'saved to {out_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='check session inputs')
    parser.add_argument('--out', type=Path, default=OUTPUT_ROOT / 'check_inputs.csv')
    main(parser.parse_args().out)
