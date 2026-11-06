'''
2026-11-06
summarise Kori's threshold-based field widths
@author: Dinghao Luo
'''

#%% imports
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matio
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.project_paths import TABLES_ROOT
WIDTH_ROOT = Path(r'Z:\Kori\PaperNov2026_CurrentBio\ReviewAnalysis\Width')
OUTPUT = ROOT / 'outputs' / 'width_kori'
FIGURES = ROOT / 'figures' / 'width_kori'
CENSOR_LIMIT = 6.99


#%% inputs
def check_order(labels, source):
    source = source.copy()
    source['neu_id'] = source['neu_id'].astype(int)
    expected = source[['rec_name', 'neu_id']].rename(columns={'rec_name': 'session'}).reset_index(drop=True)
    actual = labels[['session', 'neu_id']].reset_index(drop=True)
    if not expected.equals(actual):
        raise ValueError('Kori width order does not match the current labels')


def load_width(name):
    return loadmat(WIDTH_ROOT / name, squeeze_me=True)['fieldSizeSec'].astype(float)


def summary(frame):
    rows = []
    for (neuron_type, subset), part in frame.groupby(['neuron_type', 'subset'], sort=True):
        uncensored = part.loc[~part.censored, 'width']
        rows.append({
            'neuron_type': neuron_type,
            'subset': subset,
            'n': len(part),
            'n_censored': int(part.censored.sum()),
            'pct_censored': 100 * part.censored.mean(),
            'median': part.width.median(),
            'iqr': part.width.quantile(.75) - part.width.quantile(.25),
            'sd': part.width.std(ddof=1),
            'median_uncensored': uncensored.median() if len(uncensored) else np.nan,
            'iqr_uncensored': uncensored.quantile(.75) - uncensored.quantile(.25) if len(uncensored) else np.nan,
            'sd_uncensored': uncensored.std(ddof=1) if len(uncensored) > 1 else np.nan,
        })
    return pd.DataFrame(rows)


#%% run

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(ROOT / 'outputs' / 'fits_v1_1' / 'labels.csv')
    tic_labels = labels.loc[labels.group.isin(['tic_down', 'tic_up', 'tic_and_time_cell']), ['session', 'neu_id', 'group']].copy()
    field_labels = labels.loc[labels.is_time_cell, ['session', 'neu_id']].copy()
    field_labels['group'] = 'time_cell'

    source = matio.load_from_mat(str(TABLES_ROOT / 'RiseLastLickDownLickNeurons_thres.mat'))['RiseLLDownLickNeuronID']
    check_order(tic_labels, source)

    widths = []
    for subset, filename in {
        'all': 'TICWidthAllTr.mat',
        'early': 'TICWidthEarlyLickTr.mat',
        'late': 'TICWidthLateLickTr.mat',
    }.items():
        values = load_width(filename)
        if len(values) != len(tic_labels):
            raise ValueError(f'{filename}: width count does not match TIC labels')
        part = tic_labels.copy()
        part['neuron_type'] = part['group']
        part['subset'] = subset
        part['width'] = values
        widths.append(part[['session', 'neu_id', 'neuron_type', 'subset', 'width']])

    field_values = load_width('FieldWidthAllTr.mat')
    if len(field_values) != len(field_labels):
        raise ValueError('FieldWidthAllTr.mat: width count does not match time-cell labels')
    fields = field_labels.copy()
    fields['neuron_type'] = 'time_cell'
    fields['subset'] = 'all'
    fields['width'] = field_values
    widths.append(fields[['session', 'neu_id', 'neuron_type', 'subset', 'width']])

    table = pd.concat(widths, ignore_index=True)
    table['censored'] = table.width >= CENSOR_LIMIT
    table.to_csv(OUTPUT / 'widths.csv', index=False)
    summary(table).to_csv(OUTPUT / 'summary.csv', index=False)

    colours = {'tic_down': '#4477AA', 'tic_up': '#EE6677', 'tic_and_time_cell': '#228833', 'time_cell': '#999999'}
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    order = ['tic_down', 'tic_up', 'tic_and_time_cell', 'time_cell']
    for ax, subset in zip(axes, ['all', 'early', 'late']):
        selected = table[(table.subset == subset) | ((table.neuron_type == 'time_cell') & (subset == 'all'))]
        positions = []
        labels_for_axis = []
        for index, group in enumerate(order, start=1):
            values = selected.loc[selected.neuron_type == group, 'width'].dropna().to_numpy()
            if len(values):
                ax.boxplot(values, positions=[index], widths=.55, showfliers=False,
                           patch_artist=True, boxprops={'facecolor': colours[group], 'alpha': .5},
                           medianprops={'color': 'black'}, whiskerprops={'color': colours[group]},
                           capprops={'color': colours[group]})
                censored = selected[(selected.neuron_type == group) & selected.censored]
                if len(censored):
                    ax.scatter(np.full(len(censored), index), censored.width, color='black', marker='^', s=12, zorder=3)
                positions.append(index)
                labels_for_axis.append(group.replace('_', ' '))
        ax.set_xticks(positions, labels_for_axis, rotation=35, ha='right')
        ax.set_title(f'{subset} lick')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axhline(CENSOR_LIMIT, color='black', linestyle=':', linewidth=.7)
    axes[0].set_ylabel('threshold width (s)')
    fig.tight_layout()
    fig.savefig(FIGURES / 'width_kori.png', dpi=250)
    plt.close(fig)

    report = {
        'source_width_rule': 'Kori threshold baseline plus 0.5 SD, with low-pass filtering below 1 Hz',
        'censor_limit_seconds': CENSOR_LIMIT,
        'width_files': ['TICWidthAllTr.mat', 'TICWidthEarlyLickTr.mat', 'TICWidthLateLickTr.mat', 'FieldWidthAllTr.mat'],
        'output': str(OUTPUT),
    }
    (OUTPUT / 'run_summary.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()


