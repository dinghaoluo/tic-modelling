'''Created on 4 November 2026
Updated on 4 November 2026: use recomputed repository-local valid trials
build trial-binned fluorescence and source-table cell labels
@author: Dinghao Luo
'''

#%% imports
import json
import sys
import time
from pathlib import Path

import matio
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rec_list import SESSIONS_4S, session_path
from utils.matlab_inputs import Fluorescence, load_numeric
from utils.project_paths import TABLES_ROOT

#%% constants
OUTPUT = ROOT / 'outputs' / 'fits_v1'
BIN_END = np.arange(1, 141) * .05
BIN_TIME = BIN_END - .026

#%% labels
def source_labels():
    source = matio.load_from_mat(str(TABLES_ROOT / 'RiseLastLickDownLickNeurons_thres.mat'))
    tic = source['RiseLLDownLickNeuronID'].copy()
    ratios = source['RiseLLDownLickRatio']
    if len(tic) != len(ratios):
        raise ValueError('TIC identifiers and ratio rows differ')
    tic['neu_id'] = tic['neu_id'].astype(int)
    profile = matio.load_from_mat(str(TABLES_ROOT / 'FR_Prof' / 'PAC' / 'FRprofileTable_alignLastLickdffgfthres.mat'))
    ids = pd.DataFrame({'rec_name': np.asarray(profile['rec_name']).ravel(),
                        'neu_id': np.asarray(profile['neu_id']).ravel().astype(int)})
    clusters = matio.load_from_mat(str(TABLES_ROOT / 'PAC_Kmeans_noPCA_start_2' / 'KmeansResults.mat'))['clusters']
    idx = clusters['idx'][0, 0].ravel().astype(int)
    x = clusters['X'][0, 0]
    if len(idx) != len(ids) or len(idx) != len(x):
        raise ValueError('cluster labels and profile rows differ')
    if not np.array_equal(tic.to_numpy(), ids.to_numpy()):
        raise ValueError('TIC table and profile identifiers do not match in row order')
    # X contains the non-negative time profiles, unlike the centred centroids
    if np.any(x < 0) or np.any(x.sum(axis=1) <= 0) or len(np.unique(idx)) != 2:
        raise ValueError('cluster profiles cannot supply the requested centre of mass')
    com = (x * np.arange(x.shape[1])).sum(axis=1) / x.sum(axis=1)
    means = {int(k): float(np.mean(com[idx == k])) for k in np.unique(idx)}
    down = min(means, key=means.get)
    tic['cluster'] = np.where(idx == down, 'down', 'up')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'cluster_check.json').write_text(json.dumps({
        'cluster_rows': len(idx), 'profile_rows': len(ids),
        'mean_com_sample_index': means, 'down_cluster': int(down),
        'tic_ratio_rows': len(ratios),
        }, indent=2))
    return tic


#%% dataset
def main():
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'data' / 'binned').mkdir(parents=True, exist_ok=True)
    status = json.loads((OUTPUT / 'valid_trial_status.json').read_text())
    if not status['all_compared_match']:
        raise ValueError('valid-trial comparisons have not passed')
    tic = source_labels()
    labels, reports, alignment = [], [], []
    for session in SESSIONS_4S:
        folder = session_path(session)
        first = load_numeric(folder / 'FirstLick_LL.mat')['FirstLick'].ravel()
        valid_source = np.load(ROOT / 'data' / 'valid_trials' / f'{session}.npy')
        fields = load_numeric(folder / 'FieldDetectionShuffle' / 'FieldIndexShuffleLastLick99.mat')['FieldID'].ravel()
        if not np.all(valid_source == valid_source.astype(int)) or not np.all(fields == fields.astype(int)):
            raise ValueError(f'{session}: non-integer trial or neuron identifiers')
        valid = valid_source.astype(int) - 1
        fields = set(fields.astype(int))
        session_tic = tic[tic.rec_name == session].set_index('neu_id')['cluster'].to_dict()
        with Fluorescence(folder / 'dFFGF_befLastLick.mat') as reader:
            rows, samples = reader.shape(0)
            if samples != 5000 or len(first) != rows + 1:
                raise ValueError(f'{session}: unexpected sample count or FirstLick length')
            if np.any(valid < 0) or np.any(valid >= rows) or len(np.unique(valid)) != len(valid):
                raise ValueError(f'{session}: invalid source trial rows')
            if not fields.issubset(set(range(1, reader.n_neurons + 1))):
                raise ValueError(f'{session}: FieldID does not match neuron ids')
            if not set(session_tic).issubset(set(range(1, reader.n_neurons + 1))):
                raise ValueError(f'{session}: TIC table neuron id is outside the array')
            interval = first[valid + 1]
            valid = valid[(interval > 0) & (interval <= 6.998)]
            interval = first[valid + 1]
            binned = np.empty((reader.n_neurons, len(valid), len(BIN_TIME)), dtype=np.float32)
            drops = {offset: [] for offset in [-1, 0, 1]}
            for neuron in range(reader.n_neurons):
                if reader.shape(neuron) != (rows, samples):
                    raise ValueError(f'{session}: inconsistent neuron array dimensions')
                arr = reader.read(neuron, 1500, 5000)
                means = arr[valid].reshape(len(valid), 140, 25).mean(axis=2)
                means[BIN_END[None, :] > interval[:, None] + 1e-12] = np.nan
                binned[neuron] = means
                neu_id = neuron + 1
                cluster = session_tic.get(neu_id, '')
                is_tic, is_time = bool(cluster), neu_id in fields
                group = 'tic_and_time_cell' if is_tic and is_time else (
                    f'tic_{cluster}' if is_tic else ('time_cell' if is_time else 'other'))
                labels.append(dict(session=session, neu_id=neu_id, is_tic=is_tic,
                                   cluster=cluster, is_time_cell=is_time, group=group))
                if not is_tic:
                    continue
                sample_time = np.arange(3500) / 500
                for offset in drops:
                    paired = valid + 1 + offset
                    common = (valid > 0) & (valid < rows - 1)
                    r = valid[common]
                    end = first[paired[common]]
                    keep = (end >= .5) & (end <= 6.498)
                    r, end = r[keep], end[keep]
                    if not len(r):
                        continue
                    before = (sample_time[None, :] >= end[:, None] - .5) & (sample_time[None, :] < end[:, None])
                    after = (sample_time[None, :] >= end[:, None]) & (sample_time[None, :] < end[:, None] + .5)
                    before_mean = np.sum(np.where(before, arr[r], 0.), axis=1) / before.sum(axis=1)
                    after_mean = np.sum(np.where(after, arr[r], 0.), axis=1) / after.sum(axis=1)
                    drops[offset].extend((before_mean - after_mean).tolist())
            np.savez(ROOT / 'data' / 'binned' / f'{session}.npz', dff=binned,
                     T_k=interval, trial_rows=valid, t=BIN_TIME, bin_end=BIN_END)
            report = dict(session=session, neurons=reader.n_neurons, dff_rows=rows,
                          first_lick_length=len(first), source_valid_trials=len(valid_source),
                          valid_trials=len(valid), median_T_k=float(np.median(interval)),
                          last_lick_format=reader.format, first_lick_length_ok=True)
            reports.append(report)
            row = {'session': session, **{f'drop_{offset:+d}': float(np.mean(value)) for offset, value in drops.items()}}
            row['offset_zero_largest'] = row['drop_+0'] > max(row['drop_-1'], row['drop_+1'])
            alignment.append(row)
            print(report, row, flush=True)
        pd.DataFrame(reports).to_csv(OUTPUT / 'dataset_checks.csv', index=False)
        pd.DataFrame(alignment).to_csv(OUTPUT / 'alignment_checks.csv', index=False)
    pd.DataFrame(labels).to_csv(ROOT / 'data' / 'labels.csv', index=False)
    passed = sum(row['offset_zero_largest'] for row in alignment) > len(alignment) / 2
    (OUTPUT / 'dataset_status.json').write_text(json.dumps({
        'alignment_passed': passed, 'seconds': time.perf_counter() - started,
        'alignment_definition': 'mean raw dF/F before minus after first lick across TIC neuron-trials; full half-second windows',
        }, indent=2))
    if not passed:
        raise ValueError('offset zero does not give the largest drop in most sessions; stop before fitting')


if __name__ == '__main__':
    main()
