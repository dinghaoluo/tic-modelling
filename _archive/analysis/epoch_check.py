'''Created on 5 November 2026
measure held-out lick epochs after selecting tic labels on odd trials
@author: Dinghao Luo
'''

#%% imports
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import sem, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rec_list import SESSIONS_4S, session_path
from utils.matlab_inputs import Fluorescence, load_numeric
from analysis.label_stability import lick_ratio, number, session_average, table

#%% constants
OUTPUT = ROOT / 'outputs' / 'epoch_check'
FIGURES = ROOT / 'figures' / 'epoch_check'
EPOCHS = ['pre', 'early', 'mid', 'late', 'post']
GROUPINGS = {
    'held_out_group': ['odd_tic_down', 'odd_tic_up', 'odd_tic_new', 'not_tic'],
    'group': ['tic_down', 'tic_up', 'time_cell', 'tic_and_time_cell', 'other'],
    }
COMPARISONS = [('early', 'pre'), ('mid', 'pre'), ('late', 'early'), ('post', 'late')]

#%% trial measurements
def selected_rows(first, n_rows):
    '''retain original row parity rather than renumbering after the interval filter'''
    if len(first) != n_rows + 1:
        raise ValueError('FirstLick length is not fluorescence rows plus one')
    rows = np.arange(n_rows)
    keep = (first[1:] >= 3.) & (first[1:] <= 6.998)
    return rows[keep & (rows % 2 == 0)], rows[keep & (rows % 2 == 1)]


def epoch_means(last, first, intervals):
    '''mid epochs use sample times, with equal trial weights despite differing durations'''
    times = np.arange(5000) / 500 - 3
    mid_mask = (times[None, :] >= 1.) & (times[None, :] < intervals[:, None] - 1.)
    counts = mid_mask.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError('mid epoch has no samples')
    return dict(pre=float(last[:, 1250:1500].mean(axis=1).mean()),
                early=float(last[:, 1500:2000].mean(axis=1).mean()),
                mid=float((np.where(mid_mask, last, 0.).sum(axis=1) / counts).mean()),
                late=float(first[:, 1000:1500].mean(axis=1).mean()),
                post=float(first[:, 1500:1750].mean(axis=1).mean()))


def held_out_group(passes, source_tic, cluster):
    if not passes:
        return 'not_tic'
    if not source_tic:
        return 'odd_tic_new'
    if cluster not in ['down', 'up']:
        raise ValueError('source tic has no recognised cluster')
    return f'odd_tic_{cluster}'


#%% session work
def run_session(session):
    started = time.perf_counter()
    labels = pd.read_csv(ROOT / 'data' / 'labels.csv')
    labels = labels[labels.session == session].sort_values('neu_id').reset_index(drop=True)
    folder = session_path(session)
    intervals = load_numeric(folder / 'FirstLick_LL.mat')['FirstLick'].ravel()
    rows, exceptions, traces_last, traces_first = [], [], [], []
    with Fluorescence(folder / 'dFFGF_befLastLick.mat') as last, Fluorescence(folder / 'dFFGF_befLick.mat') as first:
        np.testing.assert_array_equal(labels.neu_id, np.arange(1, last.n_neurons + 1))
        if first.n_neurons != last.n_neurons:
            raise ValueError(f'{session}: different aligned neuron counts')
        n_rows, n_samples = last.shape(0)
        odd, even = selected_rows(intervals, n_rows)
        if not len(odd) or not len(even):
            raise ValueError(f'{session}: no selected odd or even rows')
        np.savez(OUTPUT / f'{session}_trials.npz', odd_rows=odd + 1, even_rows=even + 1,
                 odd_T_k=intervals[odd + 1], even_T_k=intervals[even + 1])
        for neuron in range(last.n_neurons):
            if n_samples != 5000 or last.shape(neuron) != (n_rows, 5000) or first.shape(neuron) != (n_rows, 5000):
                raise ValueError(f'{session}: inconsistent array dimensions')
            arr_last = last.read(neuron)
            ratio_last = lick_ratio(arr_last[odd].mean(axis=0))
            even_last = arr_last[even]
            traces_last.append(even_last[:, 1000:3000].mean(axis=0))
            del arr_last
            arr_first = first.read(neuron)
            ratio_first = lick_ratio(arr_first[odd].mean(axis=0))
            even_first = arr_first[even]
            traces_first.append(even_first[:, 0:2000].mean(axis=0))
            del arr_first
            row = labels.iloc[neuron].to_dict()
            passes = ratio_last > 1.5 and ratio_first < 2 / 3
            row.update(odd_ratio_last=ratio_last, odd_ratio_first=ratio_first, odd_tic=passes,
                       held_out_group=held_out_group(passes, row['is_tic'], row['cluster']),
                       n_odd=len(odd), n_even=len(even),
                       **epoch_means(even_last, even_first, intervals[even + 1]))
            if not np.isfinite([ratio_last, ratio_first]).all():
                exceptions.append(dict(neu_id=neuron + 1, issue='non-finite odd ratio; original comparison retained'))
            if not np.isfinite([row[name] for name in EPOCHS]).all():
                raise ValueError(f'{session}, neuron {neuron + 1}: non-finite epoch measurement')
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT / f'{session}.csv', index=False)
    np.savez(OUTPUT / f'{session}_even_traces.npz', neu_id=labels.neu_id.to_numpy(),
             last=np.asarray(traces_last), first=np.asarray(traces_first))
    status = dict(session=session, seconds=time.perf_counter() - started, neurons=len(rows),
                  total_rows=n_rows, kept_rows=len(odd) + len(even), odd_rows=len(odd), even_rows=len(even),
                  last_format=last.format, first_format=first.format, exceptions=exceptions)
    (OUTPUT / f'{session}_status.json').write_text(json.dumps(status, indent=2))
    print(session, 'complete in seconds', status['seconds'], flush=True)
    return status


#%% summary
def write_summary(frame):
    sections = []
    for column, groups in GROUPINGS.items():
        rows = []
        for group in groups:
            cells = frame[frame[column] == group]
            for epoch in EPOCHS:
                values = cells[epoch].to_numpy()
                med = np.median(values) if len(values) else np.nan
                iqr = np.subtract(*np.percentile(values, [75, 25])) if len(values) else np.nan
                parts = [part[epoch].to_numpy() for _, part in cells.groupby('session')]
                rows.append([group, epoch, len(cells), number(med), number(iqr),
                             session_average([np.median(part) for part in parts]),
                             session_average([np.subtract(*np.percentile(part, [75, 25])) for part in parts])])
        text = f'## {column}\n\n### A. Epoch distributions\n\n' + table(
            ['group', 'epoch', 'n', 'median dF/F', 'IQR dF/F', 'session median mean +/- SEM',
             'session IQR mean +/- SEM'], rows)
        rows, pair_records = [], []
        for group in groups:
            cells = frame[frame[column] == group]
            parts = [(session, part) for session, part in cells.groupby('session') if len(part) >= 5]
            for a, b in COMPARISONS:
                x = np.asarray([part[a].mean() for _, part in parts])
                y = np.asarray([part[b].mean() for _, part in parts])
                if len(x):
                    statistic, p = (0., 1.) if np.all(x == y) else wilcoxon(x, y, alternative='two-sided')
                else:
                    statistic, p = np.nan, np.nan
                rows.append([group, f'{a} minus {b}', len(parts), session_average(x - y),
                             number(statistic), number(p)])
                pair_records.extend([dict(grouping=column, group=group, comparison=f'{a} minus {b}',
                                          session=session, n=len(part), value_a=float(v_a), value_b=float(v_b))
                                     for (session, part), v_a, v_b in zip(parts, x, y)])
        text += '\n\n### B. Paired Wilcoxon across session means\n\n' + table(
            ['group', 'comparison', 'eligible sessions', 'session difference mean +/- SEM',
             'Wilcoxon statistic', 'two-sided p'], rows)
        pd.DataFrame(pair_records).to_csv(OUTPUT / f'{column}_session_pairs.csv', index=False)
        rows = []
        for group in groups:
            cells = frame[frame[column] == group]
            for a, b, sign in [('mid', 'pre', '>'), ('late', 'early', '<'), ('post', 'late', '<')]:
                condition = cells[a] > cells[b] if sign == '>' else cells[a] < cells[b]
                rows.append([group, f'{a} {sign} {b}', len(cells), int(condition.sum()),
                             number(100 * condition.mean() if len(cells) else np.nan)])
        text += '\n\n### C. Cell-wise epoch comparisons\n\n' + table(
            ['group', 'condition', 'n', 'count', '%'], rows)
        text += '\n\n### D. Group sizes\n\n' + table(['group', 'n'],
            [[group, int((frame[column] == group).sum())] for group in groups])
        sections.append(text)
    (OUTPUT / 'summary.md').write_text('\n\n'.join(sections) + '\n')


#%% figure
def plot_traces(frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    arrays = {key: np.concatenate([np.load(OUTPUT / f'{session}_even_traces.npz')[key]
                                  for session in SESSIONS_4S]) for key in ['last', 'first']}
    fig, axes = plt.subplots(1, 2, figsize=(10., 3.5))
    selections = [('odd-selected down', frame.held_out_group == 'odd_tic_down', '#4477AA'),
                  ('odd-selected up', frame.held_out_group == 'odd_tic_up', '#EE6677'),
                  ('time_cell', frame.group == 'time_cell', '#228833'),
                  ('other', frame.group == 'other', '#999999')]
    for ax, key, start in zip(axes, ['last', 'first'], [-1., -3.]):
        times = np.arange(2000) / 500 + start
        for label, mask, colour in selections:
            values = arrays[key][mask.to_numpy()]
            if not len(values):
                continue
            mean = values.mean(axis=0)
            error = sem(values, axis=0) if len(values) > 1 else np.full(2000, np.nan)
            ax.plot(times, mean, color=colour, label=label)
            ax.fill_between(times, mean - error, mean + error, color=colour, alpha=.2, linewidth=0)
        ax.set(xlabel=f'time from {key} lick (s)', ylabel='dF/F', xlim=(start, start + 4))
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / 'group_traces.png', dpi=180)
    plt.close(fig)


#%% main
def earlier_files():
    return {str(path.relative_to(ROOT)): [path.stat().st_size, path.stat().st_mtime_ns]
            for folder in ['outputs', 'figures'] for path in (ROOT / folder).rglob('*')
            if path.is_file() and OUTPUT not in path.parents and FIGURES not in path.parents}


def main():
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    prior = earlier_files()
    (OUTPUT / 'earlier_files_before.json').write_text(json.dumps(prior, indent=2))
    workers = min(os.cpu_count() or 1, len(SESSIONS_4S))
    print('session workers:', workers, flush=True)
    statuses = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_session, session) for session in SESSIONS_4S]
        for future in as_completed(futures):
            statuses.append(future.result())
    frame = pd.concat([pd.read_csv(OUTPUT / f'{session}.csv') for session in SESSIONS_4S], ignore_index=True)
    labels = pd.read_csv(ROOT / 'data' / 'labels.csv')
    pd.testing.assert_frame_equal(frame[labels.columns].fillna(''), labels.fillna(''), check_dtype=False)
    frame.to_csv(OUTPUT / 'cells.csv', index=False)
    write_summary(frame)
    plot_traces(frame)
    if earlier_files() != prior:
        raise ValueError('earlier output metadata changed')
    report = dict(seconds=time.perf_counter() - started, workers=workers, neurons=len(frame),
                  interpreter=sys.executable, earlier_files_unchanged=True, sessions=statuses,
                  trial_selection='3 <= FirstLick[r + 1] <= 6.998; original one-based odd/even row parity',
                  epochs='left-closed right-open raw sample windows; within-trial means then equal-weight trial means',
                  mid='last-lick array, 1 <= t < T_k - 1',
                  other_epochs='last-lick pre and early; first-lick late and post',
                  held_out_groups='source-table odd TICs split by cluster, new odd TICs, and not_tic; Kori groups retained separately',
                  wilcoxon='two-sided paired session means across cells; at least five cells in a group per session',
                  figure='even-trial neuron means, then group mean +/- SEM across cells; Kori time_cell and other may overlap odd-selected groups')
    (OUTPUT / 'run_report.json').write_text(json.dumps(report, indent=2))
    print('complete in seconds:', report['seconds'], flush=True)


if __name__ == '__main__':
    main()
