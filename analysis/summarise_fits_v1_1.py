'''Created on 5 November 2026
write the requested mean-trace summaries and specified restricted comparisons
@author: Dinghao Luo
'''

#%% imports
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rec_list import SESSIONS_4S
from analysis import summarise_fits as earlier
from analysis.summarise_fits import GROUPS, SHAPES, PARAMS, number, table, session_average, distribution_row

#%% constants
OUTPUT = ROOT / 'outputs' / 'fits_v1_1'
EXCLUDED = 'A132-20221124-02'

#%% requested repeated sections
def preference_table(frame, column='preferred'):
    rows = []
    sessions = [session for session in SESSIONS_4S if session in set(frame.session)]
    names = SHAPES if column == 'best_shape' else [*SHAPES, 'undecided', 'flat']
    for group in GROUPS:
        cells = frame[frame.group == group]
        for name in names:
            count = int((cells[column] == name).sum())
            parts = [cells[cells.session == session] for session in sessions]
            rows.append([group, name, count, number(100 * count / len(cells) if len(cells) else np.nan),
                         session_average([(part[column] == name).sum() for part in parts]),
                         session_average([100 * (part[column] == name).mean() if len(part) else np.nan for part in parts])])
    return table(['group', 'shape', 'count', '%', 'session count mean +/- SEM', 'session % mean +/- SEM'], rows)


def parameter_table(frame):
    rows = []
    for subset in ['beats_flat', 'prefers_shape']:
        for group in GROUPS:
            for column in PARAMS:
                shape = column.split('_')[0]
                mask = frame.beats_flat if subset == 'beats_flat' else frame.preferred == shape
                part = frame[(frame.group == group) & mask]
                rows.append([subset, group, column, len(part), *distribution_row(part, column)])
    return table(['subset', 'group', 'parameter (s)', 'cells', 'median', 'IQR', 'SD',
                  'session median mean +/- SEM', 'session IQR mean +/- SEM', 'session SD mean +/- SEM'], rows)


def unbounded_tau_table(frame):
    rows = []
    for subset in ['beats_flat', 'prefers_shape']:
        for group in GROUPS:
            for shape in ['decay', 'rise']:
                mask = frame.beats_flat if subset == 'beats_flat' else frame.preferred == shape
                part = frame[(frame.group == group) & mask]
                bound = (part[f'{shape}_mu_at_bound'] | part[f'{shape}_tau_at_bound']
                         | part[f'{shape}_a_at_bound'])
                percentages = []
                for _, session in part.groupby('session'):
                    flag = session[f'{shape}_mu_at_bound'] | session[f'{shape}_tau_at_bound'] | session[f'{shape}_a_at_bound']
                    percentages.append(100 * flag.mean())
                kept = part[~bound]
                rows.append([subset, group, shape, len(part), number(100 * bound.mean() if len(part) else np.nan),
                             session_average(percentages), len(kept), *distribution_row(kept, f'{shape}_tau')])
    return table(['subset', 'group', 'shape', 'cells before exclusion', '% at any shape bound',
                  'session bound % mean +/- SEM', 'cells not at a bound', 'tau median', 'tau IQR', 'tau SD',
                  'session median mean +/- SEM', 'session IQR mean +/- SEM', 'session SD mean +/- SEM'], rows)


def paired_table(frame):
    def metrics(part):
        return [100 * (part.preferred == 'gaussian').mean(),
                100 * part.preferred.isin(['decay', 'rise']).mean(),
                part.gaussian_mu.std(), part.gaussian_sigma.median(), part.decay_tau.median()]
    pairs, sessions = [], []
    for session in SESSIONS_4S:
        part = frame[frame.session == session]
        time_cells = part[part.group == 'time_cell']
        tics = part[part.group.isin(['tic_down', 'tic_up'])]
        if len(time_cells) >= 5 and len(tics) >= 5:
            pairs.append([metrics(time_cells), metrics(tics)])
            sessions.append(session)
    rows = []
    for i, metric in enumerate(['% gaussian', '% decay or rise', 'SD gaussian mu (s)', 'median gaussian sigma (s)', 'median decay tau (s)']):
        if not pairs:
            rows.append([metric, 0, 'NA', 'NA', 'NA', 'NA'])
            continue
        values = np.asarray(pairs)
        x, y = values[:, 0, i], values[:, 1, i]
        statistic, p = (0., 1.) if np.all(x == y) else wilcoxon(x, y, alternative='two-sided')
        rows.append([metric, len(pairs), session_average(x), session_average(y), number(statistic), number(p)])
    return table(['metric', 'paired sessions', 'time_cell mean +/- SEM', 'tic_down + tic_up mean +/- SEM',
                  'Wilcoxon statistic', 'two-sided p'], rows) + '\n\nSessions: ' + ', '.join(sessions) + '.'


#%% main
def main():
    earlier.OUTPUT = OUTPUT
    earlier.main()
    frame = pd.read_csv(OUTPUT / 'fits_all.csv')
    summary = (OUTPUT / 'summary.md').read_text()
    extra = '\n\n### 3C. Decay and rise tau excluding cells at a bound\n\n'
    extra += 'A cell is excluded here if mu, tau or amplitude of that fitted shape is flagged at a bound.\n\n'
    extra += unbounded_tau_table(frame)
    extra += '\n\n### 3C. Repeated using only cells beating flat\n\n'
    extra += parameter_table(frame[frame.beats_flat])
    extra += '\n\n' + unbounded_tau_table(frame[frame.beats_flat])
    summary = summary.replace('## 4. Spearman', extra + '\n\n## 4. Spearman')
    extra = '\n\n### 7C. Repeated using only cells beating flat\n\n' + paired_table(frame[frame.beats_flat])
    summary = summary.replace('## 8. Polynomials', extra + '\n\n## 8. Polynomials')
    summary += '\n\n## A. Best shape by point estimate, ignoring intervals\n\n' + preference_table(frame, 'best_shape')
    checks = pd.read_csv(OUTPUT / 'dataset_checks.csv')
    rows = []
    for session in SESSIONS_4S:
        part = frame[frame.session == session]
        trials = int(checks.loc[checks.session == session, 'valid_trials'].iloc[0])
        rows.append([session, trials, number(100 * part.beats_flat.mean()), number(part.best_shape_held_r2.median())])
    summary += '\n\n## B. Session summaries\n\n' + table(
        ['session', 'trials kept', '% cells beating flat', 'median held-out R2 of best shape'], rows)
    restricted = frame[frame.session != EXCLUDED]
    summary += '\n\n## D. Excluding ' + EXCLUDED + '\n\n### Section 1 repeated\n\n' + preference_table(restricted)
    summary += '\n\n### Section 7 repeated\n\n' + paired_table(restricted)
    (OUTPUT / 'summary.md').write_text(summary + '\n')


if __name__ == '__main__':
    main()
