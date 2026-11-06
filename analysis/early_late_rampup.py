'''
2026-11-06
early and late lick traces for ramp-up TICs
@author: Dinghao Luo
'''

#%% imports
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs' / 'early_late_rampup'
FIGURES = ROOT / 'figures' / 'early_late_rampup'

LICK_RANGES = {
    'early': (3.5, 4.5),
    'late': (5.0, 6.0),
}


#%% helpers
def trial_mask(first_lick, bounds):
    lower, upper = bounds
    return (first_lick >= lower) & (first_lick < upper)


def mean_trace(dff, neuron_index, trials):
    values = dff[neuron_index, trials, :]
    counts = np.isfinite(values).sum(axis=0)
    trace = np.divide(
        np.nansum(values, axis=0),
        counts,
        out=np.full(values.shape[1], np.nan, dtype=float),
        where=counts > 0,
    )
    return trace, counts


def summarise_trace(traces):
    traces = np.asarray(traces, dtype=float)
    count = np.isfinite(traces).sum(axis=0)
    mean = np.divide(
        np.nansum(traces, axis=0),
        count,
        out=np.full(traces.shape[1], np.nan, dtype=float),
        where=count > 0,
    )
    spread = np.nanstd(traces, axis=0, ddof=1)
    sem = np.divide(
        spread,
        np.sqrt(count),
        out=np.full(traces.shape[1], np.nan, dtype=float),
        where=count > 1,
    )
    return mean, sem, count


#%% load cell traces
def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(ROOT / 'outputs' / 'fits_v1_1' / 'labels.csv')
    labels = labels.loc[labels['group'].eq('tic_up')].copy()

    cell_traces = {name: [] for name in LICK_RANGES}
    session_traces = {name: {} for name in LICK_RANGES}
    trial_rows = []
    t = None

    for session, session_labels in labels.groupby('session', sort=True):
        data = np.load(ROOT / 'data' / 'binned_v1_1' / f'{session}.npz')
        dff = data['dff']
        first_lick = data['T_k']
        t = data['t']

        for subset, bounds in LICK_RANGES.items():
            selected_trials = trial_mask(first_lick, bounds)
            session_cell_traces = []

            for row in session_labels.itertuples(index=False):
                trace, counts = mean_trace(dff, int(row.neu_id) - 1, selected_trials)
                session_cell_traces.append(trace)
                cell_traces[subset].append(trace)

            session_traces[subset][session] = session_cell_traces
            trial_rows.append({
                'session': session,
                'subset': subset,
                'n_trials': int(selected_trials.sum()),
                'n_cells': len(session_cell_traces),
                'median_first_lick': float(np.median(first_lick[selected_trials])) if selected_trials.any() else np.nan,
                'minimum_first_lick': float(np.min(first_lick[selected_trials])) if selected_trials.any() else np.nan,
                'maximum_first_lick': float(np.max(first_lick[selected_trials])) if selected_trials.any() else np.nan,
            })

    population = {}
    for subset, traces in cell_traces.items():
        population[subset] = summarise_trace(traces)

    trial_summary = pd.DataFrame(trial_rows)
    trial_summary.to_csv(OUTPUT / 'trial_summary.csv', index=False)

    population_table = pd.DataFrame({'time': t})
    for subset, (mean, sem, count) in population.items():
        population_table[f'{subset}_mean'] = mean
        population_table[f'{subset}_sem'] = sem
        population_table[f'{subset}_cell_count'] = count
    population_table.to_csv(OUTPUT / 'population_trace.csv', index=False)

    cell_summary = []
    for subset, sessions in session_traces.items():
        for session, traces in sessions.items():
            for trace_index, trace in enumerate(traces):
                cell_summary.append({
                    'session': session,
                    'subset': subset,
                    'trace_index': trace_index,
                    'available_bins': int(np.isfinite(trace).sum()),
                })
    pd.DataFrame(cell_summary).to_csv(OUTPUT / 'cell_trace_summary.csv', index=False)

    #%% figure
    colours = {'early': '#2f6f9f', 'late': '#b44c4c'}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for subset, colour in colours.items():
        mean, sem, _ = population[subset]
        axes[0].plot(t, mean, color=colour, linewidth=2, label=f'{subset} lick')
        axes[0].fill_between(t, mean - sem, mean + sem, color=colour, alpha=0.18, linewidth=0)

        medians = trial_summary.loc[
            trial_summary['subset'].eq(subset), 'median_first_lick'
        ].dropna()
        if len(medians):
            axes[0].axvline(float(np.median(medians)), color=colour, linestyle='--', linewidth=1)

        for session, traces in session_traces[subset].items():
            session_mean, _, _ = summarise_trace(traces)
            axes[1].plot(t, session_mean, color=colour, alpha=0.3, linewidth=1)
        axes[1].plot(t, mean, color=colour, linewidth=2, label=f'{subset} lick')

    for ax in axes:
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('time from last lick (s)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('mean dF/F')
    axes[0].set_title('ramp-up TICs')
    axes[1].set_title('session means')
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / 'early_late_rampup.png', dpi=300)
    plt.close(fig)

    report = {
        'groups': ['tic_up'],
        'lick_ranges_seconds': {key: list(value) for key, value in LICK_RANGES.items()},
        'n_cells': int(len(labels)),
        'sessions': sorted(labels['session'].unique().tolist()),
        'figure': str(FIGURES / 'early_late_rampup.png'),
    }
    (OUTPUT / 'run_summary.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
