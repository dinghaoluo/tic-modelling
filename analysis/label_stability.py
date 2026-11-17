# -*- coding: utf-8 -*-
'''
Created on Thu Nov  5 16:41:55 2026

check tic label stability and split-half traces without fitting models

@author: Dinghao Luo
'''

#%% imports
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import sem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rec_list import SESSIONS_4S, session_path
from utils.matlab_inputs import Fluorescence

#%% constants
OUTPUT = ROOT / 'outputs' / 'label_stability'
GROUPS = ['tic_down', 'tic_up', 'time_cell', 'tic_and_time_cell', 'other']
PASSES = ['odd', 'even', 'both', 'drop_top']

#%% trial calculations
def lick_ratio(profile):
    '''normalise over the full trace before taking the original two half-second means'''
    with np.errstate(divide='ignore', invalid='ignore'):
        profile = (profile - profile.min()) / (profile.max() - profile.min())
        return float(profile[1500:1750].mean() / profile[1250:1500].mean())


def trial_masks(raw_dff):
    '''odd and even refer to one-based row numbers; equal top scores keep row order'''
    n_trials = len(raw_dff)
    odd = np.arange(n_trials) % 2 == 0
    n_drop = int(np.ceil(.1 * n_trials))
    score = raw_dff[:, 1250:1750].mean(axis=1)
    removed = np.argsort(-score, kind='stable')[:n_drop]
    kept = np.ones(n_trials, dtype=bool)
    kept[removed] = False
    return {'odd': odd, 'even': ~odd, 'drop_top': kept}, removed


def split_correlation(odd_profile, even_profile):
    '''Pearson correlation of the two means, binned over 0 <= t < 4 s'''
    odd = odd_profile[1500:3500].reshape(80, 25).mean(axis=1)
    even = even_profile[1500:3500].reshape(80, 25).mean(axis=1)
    if not np.isfinite(odd).all() or not np.isfinite(even).all():
        return np.nan
    if np.ptp(odd) == 0 or np.ptp(even) == 0:
        return np.nan
    return float(np.corrcoef(odd, even)[0, 1])


#%% session work
def run_session(session):
    labels = pd.read_csv(ROOT / 'outputs' / 'dataset' / 'labels.csv')
    labels = labels[labels.session == session].sort_values('neu_id').reset_index(drop=True)
    folder = session_path(session)
    rows, exceptions, removed_rows = [], [], []
    with Fluorescence(folder / 'dFFGF_befLastLick.mat') as last, Fluorescence(folder / 'dFFGF_befLick.mat') as first:
        np.testing.assert_array_equal(labels.neu_id, np.arange(1, last.n_neurons + 1))
        if first.n_neurons != last.n_neurons:
            raise ValueError(f'{session}: aligned arrays have different neuron counts')
        formats = dict(session=session, last_format=last.format, first_format=first.format,
                       neurons=last.n_neurons)
        for neuron in range(last.n_neurons):
            raw_dff = last.read(neuron)
            if raw_dff.shape[1] != 5000 or first.shape(neuron) != raw_dff.shape or len(raw_dff) < 2:
                raise ValueError(f'{session}, neuron {neuron + 1}: inconsistent arrays or insufficient rows')
            masks, removed = trial_masks(raw_dff)
            row = labels.iloc[neuron].to_dict()
            row.update(n_trials=len(raw_dff), n_odd=int(masks['odd'].sum()),
                       n_even=int(masks['even'].sum()), n_drop_top=len(removed),
                       n_after_drop_top=int(masks['drop_top'].sum()))
            removed_rows.append(removed + 1)
            profiles = {name: raw_dff[mask].mean(axis=0) for name, mask in masks.items()}
            row['split_half_r'] = split_correlation(profiles['odd'], profiles['even'])
            if not np.isfinite(row['split_half_r']):
                exceptions.append(dict(neu_id=neuron + 1, issue='undefined split-half correlation'))
            for name, profile in profiles.items():
                row[f'{name}_ratio_last'] = lick_ratio(profile)
            del raw_dff, profiles
            raw_dff = first.read(neuron)
            for name, mask in masks.items():
                row[f'{name}_ratio_first'] = lick_ratio(raw_dff[mask].mean(axis=0))
                row[f'pass_{name}'] = row[f'{name}_ratio_last'] > 1.5 and row[f'{name}_ratio_first'] < 2 / 3
                for alignment in ['last', 'first']:
                    if not np.isfinite(row[f'{name}_ratio_{alignment}']):
                        exceptions.append(dict(neu_id=neuron + 1, subset=name, alignment=alignment,
                                               issue='non-finite ratio; original threshold comparison retained'))
            row['pass_both'] = row['pass_odd'] and row['pass_even']
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT / f'{session}.csv', index=False)
    np.savez(OUTPUT / f'{session}_dropped_rows.npz', **{str(i + 1): value for i, value in enumerate(removed_rows)})
    status = dict(**formats, exceptions=exceptions)
    pd.DataFrame(exceptions, columns=['neu_id', 'subset', 'alignment', 'issue']).to_csv(OUTPUT / f'{session}_exceptions.csv', index=False)
    return status


#%% summary
def number(value):
    return f'{value:.5g}' if np.isfinite(value) else 'NA'


def session_average(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 'NA'
    return f'{number(values.mean())} +/- {number(sem(values) if len(values) > 1 else np.nan)}'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def correlation_row(cells):
    values = cells.split_half_r.dropna().to_numpy()
    return [len(cells), len(values), number(np.median(values) if len(values) else np.nan),
            number(np.subtract(*np.percentile(values, [75, 25])) if len(values) else np.nan)]


def write_tables(frame):
    rows = []
    for group in GROUPS:
        cells = frame[frame.group == group]
        parts = [cells[cells.session == session] for session in SESSIONS_4S]
        for name in PASSES:
            count = int(cells[f'pass_{name}'].sum())
            rows.append([group, name, len(cells), count, number(100 * count / len(cells) if len(cells) else np.nan),
                         session_average([part[f'pass_{name}'].sum() for part in parts]),
                         session_average([100 * part[f'pass_{name}'].mean() if len(part) else np.nan for part in parts])])
    text = '## A. TIC pass by group\n\n' + table(
        ['group', 'selection', 'n', 'pass count', '% passing', 'session pass count mean +/- SEM',
         'session % mean +/- SEM'], rows)
    text += '\n\n## B. Split-half correlation by group\n\n' + table(
        ['group', 'n', 'defined correlations', 'median r', 'IQR r'],
        [[group, *correlation_row(frame[frame.group == group])] for group in GROUPS])
    rows, correlations = [], []
    for session in SESSIONS_4S:
        for group in ['tic_down', 'tic_up']:
            cells = frame[(frame.session == session) & (frame.group == group)]
            correlations.append([session, group, *correlation_row(cells)])
            for name in PASSES:
                rows.append([session, group, name, len(cells), int(cells[f'pass_{name}'].sum()),
                             number(100 * cells[f'pass_{name}'].mean() if len(cells) else np.nan)])
    text += '\n\n## C. Session tables for tic_down and tic_up\n\n' + table(
        ['session', 'group', 'selection', 'n', 'pass count', '% passing'], rows)
    text += '\n\n' + table(['session', 'group', 'n', 'defined correlations', 'median r', 'IQR r'], correlations)
    rows = []
    for group in ['tic_down', 'tic_up']:
        for beats in [True, False]:
            cells = frame[(frame.group == group) & (frame.beats_flat == beats)]
            rows.append([group, beats, *correlation_row(cells)])
    text += '\n\n## D. Split-half correlation by beats-flat status\n\n' + table(
        ['group', 'beats_flat', 'n', 'defined correlations', 'median r', 'IQR r'], rows)
    (OUTPUT / 'summary.md').write_text(text + '\n')


#%% main
def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    workers = min(os.cpu_count() or 1, len(SESSIONS_4S))
    print('session workers:', workers, flush=True)
    statuses = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(run_session, session) for session in SESSIONS_4S]
        for future in as_completed(pending):
            statuses.append(future.result())
    frame = pd.concat([pd.read_csv(OUTPUT / f'{session}.csv') for session in SESSIONS_4S], ignore_index=True)
    labels = pd.read_csv(ROOT / 'outputs' / 'dataset' / 'labels.csv')
    pd.testing.assert_frame_equal(frame[labels.columns].fillna(''), labels.fillna(''), check_dtype=False)
    fits = pd.read_csv(ROOT / 'outputs' / 'fits' / 'fits_all.csv', usecols=['session', 'neu_id', 'beats_flat'])
    frame = frame.merge(fits, on=['session', 'neu_id'], how='left', validate='one_to_one')
    if frame.beats_flat.isna().any():
        raise ValueError('beats_flat missing for labelled neurons')
    frame.to_csv(OUTPUT / 'cells.csv', index=False)
    write_tables(frame)


if __name__ == '__main__':
    main()
