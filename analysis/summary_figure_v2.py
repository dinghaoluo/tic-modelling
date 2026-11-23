'''
Created on Fri Nov 21 2026
Updated on Fri Nov 21 2026
Updated on Mon Nov 23 2026

plot held-out model differences in the receptive-field summary figure

@author: Dinghao Luo
'''

#%% imports
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.compare_models import paired_test
from analysis.summary_figure import (COLOURS, FITS, FIGURES, GROUPS, LABELS, OUTPUT,
                                     p_label, parameter_values)


#%% saved contrasts
def load_contrasts():
    wider = ROOT / 'outputs' / 'wider_window_fits'
    sessions = sorted(path.name.replace('_wider_contrasts.csv', '')
                      for path in wider.glob('*_wider_contrasts.csv'))
    gaussian = pd.concat([pd.read_csv(wider / f'{session}_wider_contrasts.csv')
                          for session in sessions], ignore_index=True)
    gaussian = gaussian[gaussian.contrast.eq('gaussian_minus_decay') &
                        gaussian.group.isin(GROUPS)].copy()
    gaussian['session_median'] = gaussian.groupby(['session', 'group']).mean_contrast.transform('median')
    gaussian = gaussian.drop_duplicates(['session', 'group'])

    peak_dir = ROOT / 'outputs' / 'peak_position'
    polynomial = pd.concat([pd.read_csv(peak_dir / f'{session}_wider_polynomial_cells.csv')
                            for session in sessions], ignore_index=True)
    polynomial['session_median'] = polynomial.groupby(['session', 'group']).mean_contrast.transform('median')
    polynomial = polynomial.drop_duplicates(['session', 'group'])
    for frame, name in [(gaussian, 'Gaussian minus decay'), (polynomial, 'poly4 minus poly1')]:
        if frame.groupby('group').session.nunique().ne(len(sessions)).any():
            raise ValueError(f'{name}: not every group has one median per session')
    return sessions, gaussian, polynomial


def between_tests(frame):
    medians = frame.pivot(index='session', columns='group', values='session_median').loc[:, GROUPS]
    rows = []
    for left, right in [('tic_down', 'time_cell'), ('tic_down', 'tic_up'), ('tic_up', 'time_cell')]:
        statistic, p_value = paired_test(medians[left] - medians[right])
        rows.append(dict(group_a=left, group_b=right, n_sessions=len(medians),
                         wilcoxon_statistic=statistic, wilcoxon_p=p_value))
    return pd.DataFrame(rows)


def saved_between_tests():
    path = ROOT / 'outputs' / 'wider_group_difference' / 'group_tests.csv'
    saved = pd.read_csv(path)
    rows = []
    for left, right in [('tic_down', 'time_cell'), ('tic_down', 'tic_up'), ('tic_up', 'time_cell')]:
        row = saved[(saved.group_a == left) & (saved.group_b == right)]
        if len(row) != 1:
            raise ValueError(f'missing Gaussian-minus-decay test for {left} and {right}')
        rows.append(row.iloc[0][['group_a', 'group_b', 'n_sessions', 'wilcoxon_statistic', 'wilcoxon_p']])
    return pd.DataFrame(rows).reset_index(drop=True)


#%% figure
def make_figure(table, fits, gaussian, polynomial, between_a, between_b):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.size': 16, 'axes.titlesize': 19, 'axes.labelsize': 16,
                         'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 14,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'pdf.use14corefonts': False})
    fig = plt.figure(figsize=(13., 11.))
    grid = fig.add_gridspec(2, 6, height_ratios=[1., 1.15], hspace=.6, wspace=1.)
    axes = [fig.add_subplot(grid[0, :3]), fig.add_subplot(grid[0, 3:]),
            fig.add_subplot(grid[1, :2]), fig.add_subplot(grid[1, 2:4]), fig.add_subplot(grid[1, 4:])]

    def plot_difference(ax, panel, frame, title, ylabel, between):
        medians = frame.pivot(index='session', columns='group', values='session_median').loc[:, GROUPS] * 1000.
        values = medians.to_numpy(float)
        low, high = np.nanmin(values), np.nanmax(values)
        span = max(high - low, 1e-9)
        bracket_y = [high + (.22 + .22 * i) * span for i in range(3)]
        top = bracket_y[-1] + .28 * span
        for session in medians.index:
            ax.plot(np.arange(len(GROUPS)), medians.loc[session].to_numpy(float),
                    color='#BDBDBD', linewidth=.65, zorder=1)
        for i, group in enumerate(GROUPS):
            y = medians[group].to_numpy(float)
            ax.scatter(np.full(len(y), i), y, color=COLOURS[i], s=24, zorder=3)
            median = np.median(y)
            ax.plot([i - .12, i + .12], [median, median], color='black',
                    linewidth=2., zorder=4)
        for i, (left, right) in enumerate([(0, 2), (0, 1), (1, 2)]):
            row = between[(between.group_a == GROUPS[left]) &
                          (between.group_b == GROUPS[right])]
            if len(row) != 1:
                raise ValueError(f'{panel}: missing between-group p-value')
            height = bracket_y[i]
            ax.plot([left, left, right, right],
                    [height, height + .035 * span, height + .035 * span, height],
                    color='black', linewidth=.8)
            ax.text((left + right) / 2, height + .05 * span,
                    p_label(float(row.iloc[0].wilcoxon_p)), ha='center',
                    va='bottom', fontsize=15)
        ax.axhline(0., color='black', linewidth=.8, linestyle='--')
        ax.set(title=title, ylabel=ylabel, xticks=range(len(LABELS)),
               xticklabels=LABELS, ylim=(low - .10 * span, top))

    plot_difference(axes[0], 'A', gaussian, 'Gaussian vs exponential',
                    '\u0394 held-out log-likelihood\n(\u00d710\u207b\u00b3 per bin)', between_a)
    plot_difference(axes[1], 'B', polynomial, 'Linear vs curved',
                    '\u0394 held-out log-likelihood\n(\u00d710\u207b\u00b3 per bin)', between_b)

    for ax, panel, measure, title, ylabel in zip(axes[2:], ['C', 'D', 'E'],
            ['peak', 'width', 'ramp rate'], ['Peak location', 'Gaussian width', 'Time constant'],
            ['Peak location (s)', 'Gaussian width, σ (s)', 'Time constant, τ (s)']):
        values = [parameter_values(fits, group, measure, include_bound=True)[0]
                  [parameter_values(fits, group, measure)[1]].to_numpy()
                  for group in GROUPS]
        boxes = ax.boxplot(values, tick_labels=LABELS, showfliers=False, patch_artist=True,
                           medianprops={'color': 'black', 'linewidth': 1.5})
        for box, colour in zip(boxes['boxes'], COLOURS):
            box.set(facecolor=colour, alpha=.55)
        for i, group in enumerate(GROUPS):
            cells, parameter, _ = parameter_values(fits, group, measure, include_bound=True)
            bound_parameter = 'rise_mu' if parameter == 'peak_reference' else parameter
            bound = cells.loc[cells[f'{bound_parameter}_at_bound'].astype(bool), parameter].to_numpy()
            ax.scatter(i + 1 + np.linspace(-.13, .13, len(bound)), bound, marker='x',
                       color=COLOURS[i], s=40, linewidths=1.2, zorder=4)
        ax.set(title=title, ylabel=ylabel)
        ax.set_xticks([1, 2, 3], [label.replace(' TIC', '\nTIC').replace('Time cells', 'Time\ncells')
                                 for label in LABELS])
        ax.set_xlim(.65, 3.35)
        if panel in ['C', 'E']:
            data = np.concatenate(values)
            data_low = np.nanmin(data)
            data_high = np.nanmax(data)
            span = max(data_high - data_low, 1e-9)
            low, high = data_low - .05 * span, data_high
        else:
            low, high = ax.get_ylim()
            span = high - low
        pairs = [(0, 2), (1, 2), (0, 1)] if panel == 'C' else [(0, 2), (1, 2)]
        for i, (left, right) in enumerate(pairs):
            if left == 0 and right == 2:
                row = table[(table.group == LABELS[0]) & (table.measure == measure)].iloc[0]
            elif left == 1 and right == 2:
                row = table[(table.group == LABELS[1]) & (table.measure == measure)].iloc[0]
            else:
                row = table[(table.group == 'Ramp-down TIC vs Ramp-up TIC') &
                            (table.measure == measure)].iloc[0]
            height = high + (.08 + .19 * i) * span
            ax.plot([left + 1, left + 1, right + 1, right + 1],
                    [height, height + .035 * span, height + .035 * span, height],
                    color='black', linewidth=.8)
            ax.text((left + right + 2) / 2, height + .05 * span,
                    p_label(row['p for median']), ha='center', va='bottom', fontsize=15)
        if panel in ['C', 'E']:
            bracket_top = .85 if panel == 'C' else 1.0
            ax.set_ylim(low, high + bracket_top * span)
        else:
            ax.set_ylim(low, high + .6 * span)
        if measure == 'width':
            ax.legend(handles=[Line2D([0], [0], color='black', marker='x',
                       linestyle='none', label='At bound')],
                      frameon=False, loc='upper left', fontsize=14)
        if measure == 'ramp rate':
            ax.hlines(2.25, .78, 1.22, colors='black', linewidth=2.0, linestyles='-')
            ax.hlines(1.03, 1.78, 2.22, colors='black', linewidth=2.0, linestyles='--')
            ax.legend(handles=[Line2D([0], [0], color='black', linewidth=2.0,
                       linestyle='-', label='Cao et al. 2024 past'),
                               Line2D([0], [0], color='black', linewidth=2.0,
                       linestyle='--', label='Cao et al. 2024 future')],
                      frameon=False, loc='upper left', fontsize=14)
    for ax, letter in zip(axes, 'ABCDE'):
        ax.text(-.18, 1.05, letter, transform=ax.transAxes, fontsize=22,
                fontweight='bold', ha='right')
    fig.subplots_adjust(left=.12, right=.98, bottom=.12, top=.84)
    fig.savefig(FIGURES / 'receptive_field_model_summary_v2.png', dpi=300)
    fig.savefig(FIGURES / 'receptive_field_model_summary_v2.pdf')
    plt.close(fig)


#%% main
def main():
    sessions, gaussian, polynomial = load_contrasts()
    between_a = saved_between_tests()
    between_b = between_tests(polynomial)
    p_rows = []
    for panel, between in [('A', between_a), ('B', between_b)]:
        for row in between.itertuples(index=False):
            p_rows.append(dict(panel=panel, comparison=f'{row.group_a} vs {row.group_b}',
                               n_sessions=row.n_sessions,
                               wilcoxon_statistic=row.wilcoxon_statistic,
                               wilcoxon_p=row.wilcoxon_p))
    p_values = pd.DataFrame(p_rows)
    p_values.to_csv(OUTPUT / 'receptive_field_model_summary_v2_pvalues.csv', index=False)
    table = pd.read_csv(OUTPUT / 'summary_table_wider.csv')
    table = table[table['bound subset'].isna() | table['bound subset'].eq('included')]
    wider = ROOT / 'outputs' / 'wider_window_fits'
    fits = pd.concat([pd.read_csv(wider / f'{session}_wider_parameters.csv')
                      for session in sessions], ignore_index=True)
    make_figure(table, fits, gaussian, polynomial, between_a, between_b)
    print(p_values.to_string(index=False))


if __name__ == '__main__':
    main()
