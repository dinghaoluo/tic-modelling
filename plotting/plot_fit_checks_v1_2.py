'''Created on 5 November 2026
plot only the requested fitted-shape checks and parameter distributions
@author: Dinghao Luo
'''

#%% imports
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.rf_models_v1_2 import predict
from utils.trial_error_scores import first_lick_align

#%% constants
GROUPS = ['tic_down', 'tic_up', 'time_cell', 'tic_and_time_cell', 'other']
SHAPES = ['gaussian', 'decay', 'rise']
PREFERENCES = [*SHAPES, 'undecided', 'flat']
COLOURS = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#999999']
OUTPUT = ROOT / 'figures' / 'fits_v1_2'

#%% figures
def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUTPUT / name, dpi=180)
    plt.close(fig)


def main():
    frame = pd.read_csv(ROOT / 'outputs' / 'fits_v1_2' / 'fits_all.csv')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261104)
    chosen = []
    shortages = []
    classes = ['gaussian', 'decay', 'rise', 'undecided', 'gaussian_field', 'gaussian_monotonic']
    for preference in classes:
        if preference.startswith('gaussian_'):
            candidates = frame[(frame.preferred == 'gaussian') &
                               (frame.gaussian_class == preference.removeprefix('gaussian_'))]
        else:
            candidates = frame[frame.preferred == preference]
        size = min(6, len(candidates))
        if size < 6:
            shortages.append(dict(example_class=preference, available=len(candidates), plotted=size))
        selected = candidates.iloc[rng.choice(len(candidates), size, replace=False)]
        for _, cell in selected.iterrows():
            data = np.load(ROOT / 'data' / 'binned_v1_1' / f'{cell.session}.npz')
            interval = data['T_k']
            signal = data['dff'][int(cell.neu_id) - 1].astype(float)
            t = data['t']
            predictions = {}
            for name in SHAPES:
                params = np.array([cell[f'{name}_a0'], cell[f'{name}_a'], cell[f'{name}_mu'],
                                   cell[f'{name}_sigma' if name == 'gaussian' else f'{name}_tau']])
                prediction = predict(name, np.broadcast_to(t, signal.shape), interval[:, None], params)
                prediction[~np.isfinite(signal)] = np.nan
                predictions[name] = prediction
            fig, axes = plt.subplots(1, 2, figsize=(10., 3.4))
            for ax, alignment in zip(axes, ['last', 'first']):
                aligned_signal = signal if alignment == 'last' else first_lick_align(signal, t, interval)
                count = np.isfinite(aligned_signal).sum(axis=0)
                coordinate = t if alignment == 'last' else -(np.arange(len(t)) + .5) * .05
                order = np.argsort(coordinate)
                mean = np.divide(np.nansum(aligned_signal, axis=0), count,
                                 out=np.full(len(t), np.nan), where=count >= 10)
                ax.plot(coordinate[order], mean[order], color='black', label='mean dF/F')
                for name, colour in zip(SHAPES, COLOURS):
                    prediction = predictions[name] if alignment == 'last' else first_lick_align(predictions[name], t, interval)
                    fitted = np.divide(np.nansum(prediction, axis=0), count,
                                       out=np.full(len(t), np.nan), where=count >= 10)
                    ax.plot(coordinate[order], fitted[order], color=colour, label=name)
                ax.set(xlabel=f'time from {alignment} lick (s)', ylabel='dF/F')
                ax.legend(frameon=False)
            filename = f'{preference}_{cell.session}_{int(cell.neu_id)}.png'
            save(fig, filename)
            chosen.append(dict(example_class=preference, session=cell.session,
                               neu_id=int(cell.neu_id), figure=filename))
    result = ROOT / 'outputs' / 'fits_v1_2'
    (result / 'plot_cells.json').write_text(json.dumps(chosen, indent=2))
    (result / 'plot_shortages.json').write_text(json.dumps(shortages, indent=2))
    fig, ax = plt.subplots(figsize=(8., 4.))
    bottom = np.zeros(len(GROUPS))
    for preference, colour in zip(PREFERENCES, COLOURS):
        values = np.array([100 * (frame.loc[frame.group == group, 'preferred'] == preference).mean() for group in GROUPS])
        ax.bar(GROUPS, values, bottom=bottom, color=colour, label=preference)
        bottom += values
    ax.set(ylabel='cells (%)', ylim=(0, 100))
    ax.tick_params(axis='x', rotation=20)
    ax.legend(frameon=False, bbox_to_anchor=(1., 1.), loc='upper left')
    save(fig, 'preferred_shape_by_group.png')
    fig, ax = plt.subplots(figsize=(6., 4.))
    for group, colour in zip(GROUPS, COLOURS):
        cells = frame[frame.group == group]
        ax.scatter(cells.gaussian_mu, cells.gaussian_sigma, s=8, alpha=.4, color=colour, label=group)
    ax.set(xlabel='gaussian mu (s)', ylabel='gaussian sigma (s)')
    ax.legend(frameon=False)
    save(fig, 'gaussian_mu_sigma.png')
    for column in ['gaussian_mu', 'decay_tau', 'rise_tau']:
        fig, ax = plt.subplots(figsize=(6., 4.))
        edges = np.linspace(-1 if column == 'gaussian_mu' else .1, 7 if column == 'gaussian_mu' else 10, 31)
        for group, colour in zip(GROUPS, COLOURS):
            ax.hist(frame.loc[frame.group == group, column], bins=edges,
                    histtype='step', color=colour, label=group)
        ax.set(xlabel=column.replace('_', ' ') + ' (s)', ylabel='cells')
        ax.legend(frameon=False)
        save(fig, f'{column}_histogram.png')


if __name__ == '__main__':
    main()
