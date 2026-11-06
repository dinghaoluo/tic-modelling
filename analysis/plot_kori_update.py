'''
2026-11-06
combine the ramp-up and threshold-width figures for Kori
@author: Dinghao Luo
'''

#%% imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'figures' / 'kori_update'


#%% figure

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    trace = pd.read_csv(ROOT / 'outputs' / 'early_late_rampup' / 'population_trace.csv')
    widths = pd.read_csv(ROOT / 'outputs' / 'width_kori' / 'widths.csv')

    colours = {'early': '#2f6f9f', 'late': '#b44c4c'}
    groups = ['tic_down', 'tic_up', 'tic_and_time_cell', 'time_cell']
    width_colours = {'tic_down': '#4477AA', 'tic_up': '#EE6677',
                     'tic_and_time_cell': '#228833', 'time_cell': '#999999'}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for subset, colour in colours.items():
        mean = trace[f'{subset}_mean']
        sem = trace[f'{subset}_sem']
        axes[0].plot(trace.time, mean, color=colour, linewidth=2, label=f'{subset} lick')
        axes[0].fill_between(trace.time, mean - sem, mean + sem, color=colour, alpha=.18, linewidth=0)
    axes[0].axvline(0, color='black', linewidth=.8)
    axes[0].set(xlabel='time from last lick (s)', ylabel='mean dF/F', title='ramp-up TICs')
    axes[0].legend(frameon=False)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    for position, group in enumerate(groups, start=1):
        values = widths.loc[(widths.neuron_type == group) & (widths.subset == 'all'), 'width'].to_numpy()
        axes[1].boxplot(values, positions=[position], widths=.55, showfliers=False,
                        patch_artist=True, boxprops={'facecolor': width_colours[group], 'alpha': .55},
                        medianprops={'color': 'black'}, whiskerprops={'color': width_colours[group]},
                        capprops={'color': width_colours[group]})
        censored = widths.loc[(widths.neuron_type == group) & (widths.subset == 'all') & widths.censored, 'width']
        if len(censored):
            axes[1].scatter(np.full(len(censored), position), censored, color='black', marker='^', s=14, zorder=3)

    axes[1].axhline(6.99, color='black', linestyle=':', linewidth=.7)
    axes[1].set_xticks(range(1, len(groups) + 1), [group.replace('_', ' ') for group in groups], rotation=35, ha='right')
    axes[1].set(xlabel='', ylabel='threshold width (s)', title='Kori threshold widths')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT / 'kori_update.png', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
