'''
2026-11-09
summarise fitted Gaussian and exponential parameters by cell group
@author: Dinghao Luo
'''

#%% imports
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.rf_models_v1_2 import BOUNDS
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


def parameter_name(group, measure):
    if measure == 'Gaussian μ':
        return 'gaussian_mu'
    if measure == 'Gaussian σ':
        return 'gaussian_sigma'
    return 'rise_tau' if group == 'tic_up' else 'decay_tau'


def parameter_values(parameters, group, measure, include_bound=False):
    part = parameters[parameters.group.eq(group)].copy()
    parameter = parameter_name(group, measure)
    bound = f'{parameter}_at_bound'
    if not include_bound:
        part = part.loc[~part[bound].astype(bool)]
    part = part.loc[part[parameter].notna()]
    return part, parameter, 100 * parameters.loc[parameters.group.eq(group), bound].astype(bool).mean()


def bound_percentages(parameters, group, parameter):
    part = parameters[parameters.group.eq(group)]
    values = part.loc[part[f'{parameter}_at_bound'].astype(bool), parameter].dropna().to_numpy()
    if not len(values):
        return 0., 0.
    model = 'gaussian' if parameter.startswith('gaussian_') else 'rise' if parameter.startswith('rise_') else 'decay'
    limit_index = 0 if parameter.endswith('_mu') else 1
    lower_limit, upper_limit = BOUNDS[model][0][limit_index], BOUNDS[model][1][limit_index]
    lower = np.isclose(values, lower_limit, rtol=0., atol=1e-5)
    upper = np.isclose(values, upper_limit, rtol=0., atol=1e-5)
    denominator = len(part)
    return 100. * lower.sum() / denominator, 100. * upper.sum() / denominator


def paired_test(parameters, group, measure, include_bound):
    left, parameter, _ = parameter_values(parameters, group, measure, include_bound=include_bound)
    right, right_parameter, _ = parameter_values(parameters, 'time_cell', measure, include_bound=include_bound)
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
            cells_in, parameter, _ = parameter_values(parameters, group, measure, include_bound=True)
            cells_ex, _, _ = parameter_values(parameters, group, measure, include_bound=False)
            lower_pct, upper_pct = bound_percentages(parameters, group, parameter)
            row = {
                'group': LABELS[group],
                'measure': measure,
                'source_parameter': parameter,
                'n_cells_including_bounds': len(cells_in),
                'n_cells_excluding_bounds': len(cells_ex),
                'n_sessions_including_bounds': cells_in.session.nunique(),
                'n_sessions_excluding_bounds': cells_ex.session.nunique(),
                'median_including_bounds': cells_in[parameter].median(),
                'median_excluding_bounds': cells_ex[parameter].median(),
                'q25_excluding_bounds': cells_ex[parameter].quantile(.25),
                'q75_excluding_bounds': cells_ex[parameter].quantile(.75),
                'sd_excluding_bounds': cells_ex[parameter].std(ddof=1),
                'p_vs_time_cells_including_bounds': np.nan,
                'p_vs_time_cells_excluding_bounds': np.nan,
                'pct_lower_bound': lower_pct,
                'pct_upper_bound': upper_pct,
            }
            if group != 'time_cell':
                n_in, p_in, _ = paired_test(parameters, group, measure, include_bound=True)
                n_ex, p_ex, _ = paired_test(parameters, group, measure, include_bound=False)
                row['n_sessions_including_bounds'] = n_in
                row['n_sessions_excluding_bounds'] = n_ex
                row['p_vs_time_cells_including_bounds'] = p_in
                row['p_vs_time_cells_excluding_bounds'] = p_ex
            rows.append(row)
    return pd.DataFrame(rows)


#%% figure
def p_label(value):
    if not np.isfinite(value):
        return 'p = n/a'
    if value < .001:
        return 'p < 0.001'
    return f'p = {value:.3g}'


def histogram_bins(values, measure):
    if measure == 'Gaussian σ':
        return np.r_[np.arange(0., 4., .5), 4., np.inf], '≥4'
    if measure == 'Time constant τ':
        return np.r_[np.arange(0., 10., 1.), 10., np.inf], '≥10'
    return np.linspace(np.nanmin(values), np.nanmax(values), 26), None


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
        bins, final_label = histogram_bins(
            np.concatenate([
                parameter_values(parameters, group, measure, include_bound=True)[0][parameter_name(group, measure)].to_numpy()
                for group in GROUPS
            ]),
            measure,
        )
        for group in GROUPS:
            cells, parameter, _ = parameter_values(parameters, group, measure, include_bound=True)
            values = cells[parameter].to_numpy()
            counts, edges = np.histogram(values, bins=bins)
            proportions = counts / len(values)
            widths = np.diff(edges)
            widths[-1] = widths[-2] if np.isfinite(widths[-1]) else widths[-2]
            centres = edges[:-1] + widths / 2.
            axis.bar(centres, proportions, width=widths * .9, alpha=.48,
                     color=COLOURS[group], label=LABELS[group], align='center')
        axis.set_title(measure)
        axis.set_xlabel(xlabel)
        axis.set_ylabel('Proportion of cells')
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        if final_label is not None:
            tick_positions = edges[:-1] + widths / 2.
            tick_labels = [f'{value:g}' for value in tick_positions]
            tick_labels[-1] = final_label
            axis.set_xticks(tick_positions, tick_labels)
        for index, group in enumerate(GROUPS[:2]):
            row = table.loc[
                (table.group == LABELS[group]) & (table.measure == measure),
            ].iloc[0]
            axis.text(.03, .96 - .10 * index,
                      f'{LABELS[group]}: p in {p_label(row.p_vs_time_cells_including_bounds)}, '
                      f'p ex {p_label(row.p_vs_time_cells_excluding_bounds)}',
                      transform=axis.transAxes, va='top', fontsize=7)
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
