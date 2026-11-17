'''Created on 4 November 2026
summarise the requested cell and session comparisons without further analyses
@author: Dinghao Luo
'''

#%% imports
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import sem, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rec_list import SESSIONS_4S

#%% constants
OUTPUT = ROOT / 'outputs' / 'fits_v1'
GROUPS = ['tic_down', 'tic_up', 'time_cell', 'tic_and_time_cell', 'other']
SHAPES = ['gaussian', 'decay', 'rise']
PARAMS = ['gaussian_mu', 'gaussian_sigma', 'decay_mu', 'decay_tau', 'rise_mu', 'rise_tau']

#%% summaries
def number(value):
    return f'{value:.5g}' if np.isfinite(value) else 'NA'


def session_average(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 'NA'
    error = sem(values) if len(values) > 1 else np.nan
    return f'{number(np.mean(values))} +/- {number(error)}'


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.array([np.nan, np.nan, np.nan])
    return np.array([np.median(values), np.subtract(*np.percentile(values, [75, 25])),
                     np.std(values, ddof=1) if len(values) > 1 else np.nan])


def distribution_row(frame, column):
    pooled = distribution(frame[column])
    sessions = np.array([distribution(part[column]) for _, part in frame.groupby('session')])
    averages = [session_average(sessions[:, i]) for i in range(3)] if len(sessions) else ['NA'] * 3
    return [*(number(value) for value in pooled), *averages]


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def power_exponent(values):
    values = np.asarray(values)
    values = values[(values >= .35) & (values <= 7.) & np.isfinite(values)]
    if not len(values):
        return np.nan
    log_ratio = np.log(7. / .35)

    def nll(alpha):
        beta = 1 - alpha
        if abs(beta) < 1e-7:
            log_integral = np.log(log_ratio)
        elif beta * log_ratio > 50:
            log_integral = beta * log_ratio + np.log1p(-np.exp(-beta * log_ratio)) - np.log(beta)
        else:
            log_integral = np.log(np.expm1(beta * log_ratio) / beta)
        return alpha * np.log(values / .35).sum() + len(values) * log_integral

    result = minimize_scalar(nll, bracket=(-2., 4.))
    if not result.success:
        raise ValueError('truncated power-law likelihood did not converge')
    return float(result.x)


#%% report
def main():
    frame = pd.read_csv(OUTPUT / 'fits_all.csv')
    sections = []
    rows = []
    for group in GROUPS:
        cells = frame[frame.group == group]
        for preferred in [*SHAPES, 'undecided', 'flat']:
            count = int((cells.preferred == preferred).sum())
            by_session = [cells[cells.session == session] for session in SESSIONS_4S]
            counts = [(part.preferred == preferred).sum() for part in by_session]
            percentages = [100 * (part.preferred == preferred).mean() if len(part) else np.nan for part in by_session]
            rows.append([group, preferred, count, number(100 * count / len(cells)),
                         session_average(counts), session_average(percentages)])
    sections.append('## 1. Preferred shape\n\n' + table(
        ['group', 'shape', 'count', '%', 'session count mean +/- SEM', 'session % mean +/- SEM'], rows))
    rows = []
    for group in GROUPS:
        cells = frame[(frame.group == group) & (frame.preferred == 'gaussian')]
        count = int((cells.gaussian_mu < .1).sum())
        counts = []
        percentages = []
        for session in SESSIONS_4S:
            part = cells[cells.session == session]
            counts.append(int((part.gaussian_mu < .1).sum()))
            percentages.append(100 * (part.gaussian_mu < .1).mean() if len(part) else np.nan)
        rows.append([group, count, number(100 * count / len(cells) if len(cells) else np.nan),
                     session_average(counts), session_average(percentages)])
    sections.append('## 2. Gaussian-preferring cells with mu < 0.1 s\n\n' + table(
        ['group', 'count', '% of gaussian-preferring cells', 'session count mean +/- SEM', 'session % mean +/- SEM'], rows))
    rows = []
    for subset in ['beats_flat', 'prefers_shape']:
        for group in GROUPS:
            for column in PARAMS:
                shape = column.split('_')[0]
                mask = frame.beats_flat if subset == 'beats_flat' else frame.preferred == shape
                part = frame[(frame.group == group) & mask]
                rows.append([subset, group, column, len(part), *distribution_row(part, column)])
    sections.append('## 3. Fitted-parameter distributions\n\n' + table(
        ['subset', 'group', 'parameter (s)', 'cells', 'median', 'IQR', 'SD',
         'session median mean +/- SEM', 'session IQR mean +/- SEM', 'session SD mean +/- SEM'], rows))
    rows = []
    for group in GROUPS:
        cells = frame[frame.group == group]
        for shape in SHAPES:
            x, y = f'{shape}_mu', f'{shape}_sigma' if shape == 'gaussian' else f'{shape}_tau'
            rho = spearmanr(cells[x], cells[y]).statistic
            session_rho = [spearmanr(part[x], part[y]).statistic for _, part in cells.groupby('session') if len(part) > 1]
            rows.append([group, shape, len(cells), number(rho), session_average(session_rho)])
    sections.append('## 4. Spearman parameter correlations\n\n' + table(
        ['group (all cells)', 'shape', 'cells', 'rho', 'session rho mean +/- SEM'], rows))
    cells = frame[frame.group.isin(['time_cell', 'tic_and_time_cell'])]
    included = cells[(cells.gaussian_mu >= .35) & (cells.gaussian_mu <= 7.)]
    estimate = power_exponent(included.gaussian_mu.to_numpy())
    rng = np.random.default_rng(20261104)
    session_values = {session: included.loc[included.session == session, 'gaussian_mu'].to_numpy() for session in SESSIONS_4S}
    samples = [power_exponent(np.concatenate([session_values[session] for session in rng.choice(SESSIONS_4S, len(SESSIONS_4S), replace=True)])) for _ in range(1000)]
    lower, upper = np.nanpercentile(samples, [2.5, 97.5])
    sections.append('## 5. Truncated power-law exponent of time-cell peaks\n\n' + table(
        ['groups', 'mu range (s)', 'included cells', 'alpha', 'session-bootstrap 95% interval', 'session alpha mean +/- SEM'],
        [['time_cell + tic_and_time_cell', '0.35 to 7', len(included), number(estimate),
          f'{number(lower)} to {number(upper)}', session_average([power_exponent(value) for value in session_values.values()])]]))
    rows = []
    for group in GROUPS:
        for shape in SHAPES:
            cells = frame[(frame.group == group) & (frame.preferred == shape)]
            med, iqr, _ = distribution(cells.rho)
            significant = 100 * (cells.scaling_ll_gain_lower > 0).mean() if len(cells) else np.nan
            session_medians, session_iqrs, session_sig = [], [], []
            for _, part in cells.groupby('session'):
                values = distribution(part.rho)
                session_medians.append(values[0])
                session_iqrs.append(values[1])
                session_sig.append(100 * (part.scaling_ll_gain_lower > 0).mean())
            rows.append([group, shape, len(cells), number(med), number(iqr), number(significant),
                         session_average(session_medians), session_average(session_iqrs), session_average(session_sig)])
    sections.append('## 6. Interval scaling\n\n' + table(
        ['group', 'preferred shape', 'cells', 'rho median', 'rho IQR', '% gain interval above zero',
         'session median mean +/- SEM', 'session IQR mean +/- SEM', 'session % mean +/- SEM'], rows))
    def metrics(part):
        return [100 * (part.preferred == 'gaussian').mean(),
                100 * part.preferred.isin(['decay', 'rise']).mean(),
                part.gaussian_mu.std(), part.gaussian_sigma.median(), part.decay_tau.median()]
    pairs = []
    eligible = []
    for session in SESSIONS_4S:
        part = frame[frame.session == session]
        time_cells = part[part.group == 'time_cell']
        tics = part[part.group.isin(['tic_down', 'tic_up'])]
        if len(time_cells) >= 5 and len(tics) >= 5:
            pairs.append([metrics(time_cells), metrics(tics)])
            eligible.append(session)
    pairs = np.asarray(pairs)
    rows = []
    for i, name in enumerate(['% gaussian', '% decay or rise', 'SD gaussian mu (s)', 'median gaussian sigma (s)', 'median decay tau (s)']):
        x, y = pairs[:, 0, i], pairs[:, 1, i]
        if np.all(x == y):
            statistic, p = 0., 1.
        else:
            result = wilcoxon(x, y, alternative='two-sided')
            statistic, p = result.statistic, result.pvalue
        rows.append([name, len(pairs), session_average(x), session_average(y), number(statistic), number(p)])
    sections.append('## 7. Paired session comparisons\n\n' + table(
        ['metric', 'paired sessions', 'time_cell mean +/- SEM', 'tic_down + tic_up mean +/- SEM', 'Wilcoxon statistic', 'two-sided p'], rows)
        + '\n\nSessions: ' + ', '.join(eligible) + '.')
    rows = []
    for group in GROUPS:
        cells = frame[frame.group == group]
        for degree in range(1, 5):
            count = int((cells.best_poly_degree == degree).sum())
            parts = [part for _, part in cells.groupby('session')]
            rows.append([group, degree, count, number(100 * count / len(cells)),
                         session_average([(part.best_poly_degree == degree).sum() for part in parts]),
                         session_average([100 * (part.best_poly_degree == degree).mean() for part in parts])])
    text = table(['group', 'best degree', 'count', '%', 'session count mean +/- SEM', 'session % mean +/- SEM'], rows)
    rows = []
    for group in GROUPS:
        cells = frame[frame.group == group]
        for label, column in [('best polynomial', 'best_poly_held_r2'), ('best shape', 'best_shape_held_r2')]:
            rows.append([group, label, number(cells[column].median()),
                         session_average([part[column].median() for _, part in cells.groupby('session')])])
    sections.append('## 8. Polynomials\n\n' + text + '\n\n' + table(
        ['group', 'model', 'median held-out R2', 'session median mean +/- SEM'], rows))
    (OUTPUT / 'summary.md').write_text('\n\n'.join(sections) + '\n')


if __name__ == '__main__':
    main()

