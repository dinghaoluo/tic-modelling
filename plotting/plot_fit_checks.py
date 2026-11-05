'''Created on 4 November 2026
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
from utils.rf_models import predict

#%% constants
GROUPS = ['tic_down', 'tic_up', 'time_cell', 'tic_and_time_cell', 'other']
SHAPES = ['gaussian', 'decay', 'rise']
PREFERENCES = [*SHAPES, 'undecided', 'flat']
COLOURS = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#999999']
OUTPUT = ROOT / 'figures' / 'fits_v1'

#%% figures
def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUTPUT / name, dpi=180)
    plt.close(fig)


def main():
    frame = pd.read_csv(ROOT / 'outputs' / 'fits_v1' / 'fits_all.csv')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261104)
    chosen = []
    for preference in ['gaussian', 'decay', 'rise', 'undecided']:
        candidates = frame[frame.preferred == preference]
        if len(candidates) < 6:
            raise ValueError(f'fewer than six {preference} cells; cannot draw the requested sample')
        selected = candidates.iloc[rng.choice(len(candidates), 6, replace=False)]
        for _, cell in selected.iterrows():
            data = np.load(ROOT / 'data' / 'binned' / f'{cell.session}.npz')
            interval = data['T_k']
            mask = (interval >= 3.5) & (interval <= 4.5)
            if not mask.any():
                raise ValueError(f'{cell.session}: no trials in the requested example interval')
            signal = data['dff'][int(cell.neu_id) - 1, mask]
            t = data['t']
            count = np.isfinite(signal).sum(axis=0)
            mean = np.divide(np.nansum(signal, axis=0), count,
                             out=np.full(len(t), np.nan), where=count > 0)
            fig, ax = plt.subplots(figsize=(5., 3.4))
            ax.plot(t, mean, color='black', label='mean dF/F')
            for name, colour in zip(SHAPES, COLOURS):
                params = np.array([cell[f'{name}_a0'], cell[f'{name}_a'], cell[f'{name}_mu'],
                                   cell[f'{name}_sigma' if name == 'gaussian' else f'{name}_tau']])
                prediction = predict(name, np.broadcast_to(t, signal.shape), interval[mask, None], params)
                prediction[~np.isfinite(signal)] = np.nan
                fitted = np.divide(np.nansum(prediction, axis=0), count,
                                   out=np.full(len(t), np.nan), where=count > 0)
                ax.plot(t, fitted, color=colour, label=name)
            ax.set(xlabel='time from last lick (s)', ylabel='dF/F')
            ax.legend(frameon=False)
            filename = f'{preference}_{cell.session}_{int(cell.neu_id)}.png'
            save(fig, filename)
            chosen.append(dict(preferred=preference, session=cell.session,
                               neu_id=int(cell.neu_id), figure=filename))
    (ROOT / 'outputs' / 'fits_v1' / 'plot_cells.json').write_text(json.dumps(chosen, indent=2))
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
        edges = np.linspace(0, 7 if column == 'gaussian_mu' else 10, 31)
        for group, colour in zip(GROUPS, COLOURS):
            ax.hist(frame.loc[frame.group == group, column], bins=edges,
                    histtype='step', color=colour, label=group)
        ax.set(xlabel=column.replace('_', ' ') + ' (s)', ylabel='cells')
        ax.legend(frameon=False)
        save(fig, f'{column}_histogram.png')


if __name__ == '__main__':
    main()
