'''
2026-11-09
summarise fitted Gaussian and exponential parameters by cell group
@author: Dinghao Luo
'''

#%% imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
WIDER = ROOT / 'outputs' / 'undecided_check'
OUTPUT = ROOT / 'outputs' / 'wider_evidence'
FIGURES = ROOT / 'figures' / 'wider_evidence'

GROUPS = ['tic_down', 'tic_up', 'time_cell']
LABELS = {
    'tic_down': 'Ramp-down TIC',
    'tic_up': 'Ramp-up TIC',
    'time_cell': 'Time cells',
}
COLOURS = {
    'tic_down': '#228833',
    'tic_up': '#4477AA',
    'time_cell': '#9467BD',
}


#%% summaries
def load_parameters():
    paths = sorted(WIDER.glob('*_wider_parameters.csv'))
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def parameter_values(parameters, group, measure, include_bound=False):
    part = parameters[parameters.group.eq(group)].copy()
    if measure == 'Gaussian μ':
        parameter = 'gaussian_mu'
    elif measure == 'Gaussian σ':
        parameter = 'gaussian_sigma'
    else:
        parameter = 'rise_tau' if group == 'tic_up' else 'decay_tau'
    bound = f'{parameter}_at_bound'
    if not include_bound:
        part = part.loc[~part[bound].astype(bool)]
    part = part.loc[part[parameter].notna()]
    return part, parameter, 100 * parameters.loc[parameters.group.eq(group), bound].astype(bool).mean()


def paired_test(parameters, group, measure):
    left, parameter, _ = parameter_values(parameters, group, measure)
    right, right_parameter, _ = parameter_values(parameters, 'time_cell', measure)
    pairs = []
    for session in sorted(set(left.session) & set(right.session)):
        left_values = left.loc[left.session.eq(session), parameter]
        right_values = right.loc[right.session.eq(session), right_parameter]
        if len(left_values) and len(right_values):
            pairs.append((left_values.median(), right_values.median()))
    pairs = np.asarray(pairs, dtype=float)
    differences = pairs[:, 0] - pairs[:, 1]
    return len(pairs), float(wilcoxon(differences).pvalue), pairs


def summary_table(parameters):
    rows = []
    for group in GROUPS:
        for measure in ['Gaussian μ', 'Gaussian σ', 'Time constant τ']:
            actual_measure = 'Time constant τ' if measure == 'Time constant τ' else measure
            cells, parameter, pct_at_bound = parameter_values(parameters, group, actual_measure)
            row = {
                'group': LABELS[group],
                'measure': measure,
                'source_parameter': parameter,
                'n_cells': len(cells),
                'n_sessions': cells.session.nunique(),
                'median': cells[parameter].median(),
                'q25': cells[parameter].quantile(.25),
                'q75': cells[parameter].quantile(.75),
                'sd': cells[parameter].std(ddof=1),
                'p_vs_time_cells': np.nan,
                'pct_at_bound': pct_at_bound,
            }
            if group != 'time_cell':
                n_sessions, p_value, _ = paired_test(parameters, group, actual_measure)
                row['n_sessions'] = n_sessions
                row['p_vs_time_cells'] = p_value
            rows.append(row)
    return pd.DataFrame(rows)


#%% figure
def p_label(value):
    if not np.isfinite(value):
        return 'p = n/a'
    if value < .001:
        return 'p < 0.001'
    return f'p = {value:.3g}'


def make_figure(parameters, table):
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'pdf.use14corefonts': False,
        'font.size': 9,
        'axes.titlesize': 11,
        'axes.labelsize': 9,
    })
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), constrained_layout=True)
    measures = [
        ('Gaussian μ', 'μ (s)'),
        ('Gaussian σ', 'σ (s)'),
        ('Time constant τ', 'τ (s)'),
    ]
    for axis, (measure, xlabel) in zip(axes, measures):
        pieces = []
        bounds = []
        for group in GROUPS:
            cells, parameter, _ = parameter_values(parameters, group, measure, include_bound=True)
            unbound, _, _ = parameter_values(parameters, group, measure)
            pieces.append(cells[parameter].to_numpy())
            bounds.append(cells.loc[~cells.index.isin(unbound.index), parameter].to_numpy())
        all_values = np.concatenate(pieces)
        bins = np.linspace(np.nanmin(all_values), np.nanmax(all_values), 26)
        for group, values in zip(GROUPS, pieces):
            axis.hist(values, bins=bins, density=True, alpha=.48,
                      color=COLOURS[group], label=LABELS[group])
        axis.set_title(measure)
        axis.set_xlabel(xlabel)
        axis.set_ylabel('Density')
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        if measure != 'Gaussian μ':
            axis.set_xlim(bins[0], bins[-1])
        for index, group in enumerate(GROUPS[:2]):
            p_value = table.loc[
                (table.group == LABELS[group]) & (table.measure == measure),
                'p_vs_time_cells',
            ].iloc[0]
            axis.text(.03, .96 - .09 * index,
                      f'{LABELS[group]}: {p_label(p_value)}',
                      transform=axis.transAxes, va='top', fontsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    for label, axis in zip('ABC', axes):
        axis.text(-.10, 1.06, label, transform=axis.transAxes,
                  fontsize=12, fontweight='bold', va='top')
    fig.savefig(FIGURES / 'parameter_distributions.png', dpi=300)
    fig.savefig(FIGURES / 'parameter_distributions.pdf')
    plt.close(fig)


#%% run
def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    parameters = load_parameters()
    table = summary_table(parameters)
    table.to_csv(OUTPUT / 'wider_parameter_summary.csv', index=False)
    (OUTPUT / 'wider_parameter_summary.md').write_text(
        '# fitted parameter distributions\n\n' + table.to_string(index=False) + '\n',
        encoding='utf-8',
    )
    make_figure(parameters, table)


if __name__ == '__main__':
    main()
