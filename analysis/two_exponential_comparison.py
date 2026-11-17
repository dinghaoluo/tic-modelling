# -*- coding: utf-8 -*-
'''
Created on Mon Nov 16 16:48:27 2026

plot saved two-exponential model comparisons

@author: Dinghao Luo
'''

#%% imports
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'outputs' / 'two_exponential' / 'comparison_summary.csv'
FIGURES = ROOT / 'figures'
GROUPS = ['tic_down', 'tic_up', 'time_cell']
LABELS = ['Ramp-down TIC', 'Ramp-up TIC', 'Time cells']
COLOURS = ['#4D4D4D', '#E69F00', '#BDBDBD']
PRIMARY = {'tic_down': 'last', 'tic_up': 'next', 'time_cell': 'last'}
EXPONENTIAL = {'tic_down': 'decay', 'tic_up': 'rise', 'time_cell': 'decay'}

#%% saved summaries
def p_label(p_value):
    if p_value < .001:
        return 'p < 0.001'
    return f'p = {p_value:#.2g}' + (' (n.s.)' if p_value >= .05 else '')


def selected_rows(table, models):
    rows = []
    for group in GROUPS:
        subset = table[(table.group == group) & (table.model_a == PRIMARY[group]) &
                       (table.model_b == models[group]) & (table.scoring == 'all')]
        if len(subset) != 1:
            raise ValueError(f'expected one saved all-bin row for {group}, {models[group]}')
        rows.append(subset.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)

#%% figure
def make_figure(table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 16, 'axes.titlesize': 19, 'axes.labelsize': 16,
                         'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 14,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'pdf.use14corefonts': False})
    fig, axes = plt.subplots(1, 2, figsize=(13., 5.8), sharey=True)
    panels = [('A', 'Two-exponential vs group exponential', EXPONENTIAL),
              ('B', 'Two-exponential vs Gaussian', {group: 'gaussian' for group in GROUPS})]
    annotations = []
    for ax, (panel, title, models) in zip(axes, panels):
        selected = selected_rows(table, models)
        bottom = np.zeros(len(GROUPS))
        for field, colour, label in [('better', COLOURS[0], 'Two-exponential better'),
                                     ('worse', COLOURS[1], 'Other model better'),
                                     ('undecided', COLOURS[2], 'No clear difference')]:
            values = selected[field].to_numpy(float)
            ax.bar(np.arange(len(GROUPS)), values, bottom=bottom, color=colour,
                   width=.65, label=label)
            bottom += values
        np.testing.assert_allclose(bottom, 100., atol=1e-10)
        for i, row in selected.iterrows():
            text = p_label(float(row.wilcoxon_p))
            ax.text(i, 103., text, ha='center', va='bottom', fontsize=15)
            annotations.append({'panel': panel, 'group': LABELS[i],
                                'two_exponential_better': float(row.better),
                                'other_model_better': float(row.worse),
                                'no_clear_difference': float(row.undecided),
                                'p': float(row.wilcoxon_p), 'p_label': text})
        ax.set(title=title, ylabel='Cells (%)', ylim=(0., 114.), yticks=np.arange(0., 101., 20.),
               xticks=range(len(LABELS)), xticklabels=LABELS)
        ax.text(-.12, 1.05, panel, transform=ax.transAxes, fontsize=22,
                fontweight='bold', ha='right')
    axes[1].set_ylabel('')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='upper center', bbox_to_anchor=(.5, .995),
               ncol=3, fontsize=14, handlelength=1., columnspacing=1.)
    fig.subplots_adjust(left=.085, right=.98, bottom=.17, top=.78, wspace=.28)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / 'two_exponential_comparison.png', dpi=300)
    fig.savefig(FIGURES / 'two_exponential_comparison.pdf')
    plt.close(fig)
    return annotations

#%% main
def main():
    table = pd.read_csv(INPUT)
    annotations = make_figure(table)
    print(pd.DataFrame(annotations).to_string(index=False))


if __name__ == '__main__':
    main()
