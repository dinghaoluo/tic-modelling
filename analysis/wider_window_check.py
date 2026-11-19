'''
Created on Thu Nov  5 18:24:50 2026
Updated on Wed Nov 18 2026

compare the current interval window with an added pre-last-lick period

@author: Dinghao Luo
'''

#%% imports
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rec_list import SESSIONS_4S, session_path
from utils.matlab_inputs import Fluorescence
from utils.rf_models import BOUNDS, fit_field
from analysis.fit_shapes import cv_prediction
from analysis.compare_models_trial import CONTRASTS, PRIMARY, summary_row
from analysis.compare_models import EXCLUDED
from analysis.recovery_check import (OUTPUT, FIGURES, FITS, GROUPS, LABELS, matched_shape,
                                     scored_contrast, sample_cells, recovery_session, recovery_summary)

#%% saved wider-window inputs
def read_wider_inputs(session, saved):
    with np.load(ROOT / 'data' / 'binned_wider' / f'{session}.npz') as source:
        data = dict(source)
    with np.load(FITS / f'{session}_folds.npz') as source:
        folds = dict(source)
    np.testing.assert_array_equal(data['neu_id'], saved['neu_id'][:])
    np.testing.assert_array_equal(data['trial_rows'], folds['trial_rows'])
    np.testing.assert_array_equal(data['trial_rows'] + 1, saved['trial_rows'][:])
    np.testing.assert_array_equal(data['t'], saved['t'][:])
    n_trials = len(data['T_k'])
    np.testing.assert_array_equal(np.unique(folds['folds']), np.arange(5))
    bootstrap = folds['bootstrap_trials']
    if (data['dff'].shape != saved['flat'].shape or folds['folds'].shape != (n_trials,)
            or bootstrap.shape[1] != n_trials or bootstrap.min() < 0 or bootstrap.max() >= n_trials):
        raise ValueError(f'{session}: saved predictions, folds or resamples do not match the trials')
    after_lick = data['bin_end'][None, :] > data['T_k'][:, None] + 1e-12
    if np.isfinite(data['dff'][:, after_lick]).any():
        raise ValueError(f'{session}: observed bins extend beyond the next lick')
    cells = pd.read_csv(OUTPUT / f'{session}_wider_parameters.csv')
    np.testing.assert_array_equal(cells.neu_id, data['neu_id'])
    return data, folds, cells


#%% wider-window fits
def wider_session(session):
    fits = pd.read_csv(FITS / 'fits_all.csv')
    fits = fits[fits.session == session].reset_index(drop=True)
    old = np.load(ROOT / 'data' / 'binned' / f'{session}.npz')
    np.testing.assert_array_equal(fits.neu_id, np.arange(1, len(old['dff']) + 1))
    selected = fits[fits.group.isin(GROUPS)].copy()
    folds = np.load(FITS / f'{session}_folds.npz')
    np.testing.assert_array_equal(folds['trial_rows'], old['trial_rows'])
    intervals, trial_rows = old['T_k'], old['trial_rows']
    bin_end = np.r_[np.arange(-19, 1) * .05, old['bin_end']]
    times = np.r_[np.arange(-20, 0) * .05 + .024, old['t']]
    binned = np.empty((len(selected), len(intervals), len(times)), dtype=np.float32)
    with Fluorescence(session_path(session) / 'dFFGF_befLastLick.mat') as reader:
        for i, cell in enumerate(selected.itertuples()):
            raw_dff = reader.read(int(cell.neu_id) - 1, 1000, 5000)
            values = raw_dff[trial_rows].reshape(len(intervals), 160, 25).mean(axis=2)
            values[bin_end[None, :] > intervals[:, None] + 1e-12] = np.nan
            binned[i] = values
            np.testing.assert_array_equal(binned[i, :, 20:], old['dff'][int(cell.neu_id) - 1])
    np.savez(ROOT / 'data' / 'binned_wider' / f'{session}.npz', dff=binned, T_k=intervals,
             trial_rows=trial_rows, t=times, bin_end=bin_end, neu_id=selected.neu_id.to_numpy())
    full_rows, rows, failures = [], [], []
    with h5py.File(OUTPUT / f'{session}_wider_oof.h5', 'w') as saved:
        saved.create_dataset('neu_id', data=selected.neu_id.to_numpy())
        saved.create_dataset('trial_rows', data=trial_rows + 1)
        saved.create_dataset('t', data=times)
        for name in ['flat', 'gaussian', 'decay', 'rise']:
            saved.create_dataset(name, shape=binned.shape, dtype='f8', chunks=(1, len(intervals), len(times)),
                                 compression='lzf', fillvalue=np.nan)
        for i, (_, cell) in enumerate(selected.iterrows()):
            signal = binned[i].astype(float)
            predictions, cell_failures = {}, []
            param_row = dict(session=session, neu_id=int(cell.neu_id), group=cell.group)
            trial, bins = np.where(np.isfinite(signal))
            for name in ['flat', 'gaussian', 'decay', 'rise']:
                predictions[name], failed = cv_prediction(name, signal, times, intervals, folds['folds'])
                saved[name][i] = predictions[name]
                cell_failures.extend(failed)
                if name == 'flat':
                    continue
                params, success = fit_field(name, times[bins], intervals[trial], signal[trial, bins])
                if not success:
                    cell_failures.append(dict(model=name, stage='full'))
                width = 'sigma' if name == 'gaussian' else 'tau'
                names = ['a0', 'a', 'mu', width]
                for parameter, value in zip(names, params):
                    param_row[f'{name}_{parameter}'] = value
                lower, upper = BOUNDS[name]
                flags = np.isclose(params[2:], lower, rtol=0, atol=1e-5) | np.isclose(params[2:], upper, rtol=0, atol=1e-5)
                for parameter, flag in zip(['mu', width], flags):
                    param_row[f'{name}_{parameter}_at_bound'] = bool(flag)
                param_row[f'{name}_a_at_bound'] = bool(params[1] <= 1e-5)
            for model_a, model_b in CONTRASTS:
                result, errors = scored_contrast(signal, predictions, model_a, model_b, folds['bootstrap_trials'])
                rows.append(dict(session=session, neu_id=int(cell.neu_id), group=cell.group,
                                 contrast=f'{model_a}_minus_{model_b}', model_a=model_a, model_b=model_b,
                                 primary=(cell.group, f'{model_a}_minus_{model_b}') in PRIMARY, **result))
            param_row['fit_success'] = not cell_failures
            full_rows.append(param_row)
            failures.extend([dict(neu_id=int(cell.neu_id), **failure) for failure in cell_failures])
            if len(full_rows) % 20 == 0:
                print(session, 'wider cells', len(full_rows), flush=True)
    pd.DataFrame(rows).to_csv(OUTPUT / f'{session}_wider_contrasts.csv', index=False)
    pd.DataFrame(full_rows).to_csv(OUTPUT / f'{session}_wider_parameters.csv', index=False)
    status = dict(session=session, failures=failures,
                  cells=len(selected), source='dFFGF_befLastLick.mat', current_window_identical=True)
    pd.DataFrame([dict(failure, session=session) for failure in failures], columns=['session', 'neu_id', 'model', 'fold', 'stage']).to_csv(OUTPUT / f'{session}_wider_failures.csv', index=False)
    return status


def run_both(session):
    recovery = recovery_session(session)
    wider = wider_session(session)
    return dict(session=session, recovery=recovery, wider=wider)


#%% old and new summaries
def population(frame):
    rows = []
    for subset, selected in [('all_sessions', frame),
                             ('excluded_sessions', frame[~frame.session.isin(EXCLUDED)])]:
        for group in GROUPS:
            for model_a, model_b in CONTRASTS:
                contrast = f'{model_a}_minus_{model_b}'
                cells = selected[(selected.group == group) & (selected.contrast == contrast)]
                common = dict(subset=subset, group=group, contrast=contrast, model_a=model_a, model_b=model_b,
                              primary=(group, contrast) in PRIMARY)
                rows.append(dict(**common, level='group', session='', **summary_row(cells)))
                for session, part in cells.groupby('session'):
                    values = summary_row(part)
                    values.update(wilcoxon_statistic=np.nan, wilcoxon_p=np.nan)
                    rows.append(dict(**common, level='session', session=session, **values))
    return pd.DataFrame(rows)


def wider_summary():
    wider = pd.concat([pd.read_csv(OUTPUT / f'{session}_wider_contrasts.csv') for session in SESSIONS_4S], ignore_index=True)
    params = pd.concat([pd.read_csv(OUTPUT / f'{session}_wider_parameters.csv') for session in SESSIONS_4S], ignore_index=True)
    old = pd.read_csv(ROOT / 'outputs' / 'model_comparison' / 'cell_ll_trial.csv')
    old = old[old.group.isin(GROUPS)]
    columns = ['session', 'neu_id', 'group', 'contrast']
    merged = wider.merge(old[columns + ['mean_contrast', 'lower', 'upper', 'decision']], on=columns,
                         how='left', validate='one_to_one', suffixes=('', '_old'))
    if merged.decision_old.isna().any():
        raise ValueError('missing old decision for wider cell')
    merged['decision_changed'] = merged.decision != merged.decision_old
    merged = merged.merge(params, on=['session', 'neu_id', 'group'], validate='many_to_one')
    merged.to_csv(OUTPUT / 'wider_window_cells.csv', index=False)
    all_rows = []
    current_population = population(old)
    wider_population = population(wider)
    for window, values in [('current', current_population), ('wider', wider_population)]:
        values = values.copy()
        values['window'] = window
        values['record_type'] = 'contrast'
        all_rows.append(values)
    keys = ['subset', 'group', 'contrast', 'level', 'session']
    side = current_population[current_population.primary].merge(wider_population[wider_population.primary],
            on=keys, suffixes=('_old', '_new'), validate='one_to_one')
    side.to_csv(OUTPUT / 'primary_window_comparison.csv', index=False)
    fit_old = pd.read_csv(FITS / 'fits_all.csv')
    parameter_rows, transition_rows = [], []
    for subset in ['all_sessions', 'excluded_sessions']:
        for window, values in [('current', fit_old), ('wider', params)]:
            part = values if subset == 'all_sessions' else values[~values.session.isin(EXCLUDED)]
            for group in GROUPS:
                cells = part[part.group == group]
                for parameter in ['gaussian_mu', 'gaussian_sigma', 'decay_mu', 'decay_tau', 'rise_mu', 'rise_tau']:
                    parameter_rows.append(dict(record_type='parameter', subset=subset, window=window, group=group,
                                               parameter=parameter, n_cells=len(cells), median=cells[parameter].median(),
                                               pct_at_bound=100 * cells[f'{parameter}_at_bound'].mean()))
        part = merged if subset == 'all_sessions' else merged[~merged.session.isin(EXCLUDED)]
        for (group, contrast), cells in part.groupby(['group', 'contrast']):
            for (before, after), chunk in cells.groupby(['decision_old', 'decision']):
                transition_rows.append(dict(record_type='decision_transition', subset=subset, group=group,
                                            contrast=contrast, decision_old=before, decision_new=after,
                                            count=len(chunk), n_cells=len(cells), percentage=100 * len(chunk) / len(cells)))
    all_rows += [pd.DataFrame(parameter_rows), pd.DataFrame(transition_rows)]
    pd.concat(all_rows, ignore_index=True).to_csv(OUTPUT / 'wider_window_summary.csv', index=False)
    return merged


#%% figure
def plot_checks():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from analysis.summary_figure import p_label

    recovery = pd.read_csv(OUTPUT / 'recovery_summary.csv')
    summary = pd.read_csv(OUTPUT / 'wider_window_summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(14., 5.))
    palette = ['#4477AA', '#EE6677', '#BBBBBB']
    coordinates, labels = [], []
    for i, (group, label) in enumerate(zip(GROUPS, LABELS)):
        for j, truth in enumerate(['gaussian', matched_shape(group)]):
            position = 3 * i + j
            bottom = 0.
            for outcome, colour in zip(['correct', 'wrong', 'undecided'], palette):
                row = recovery[(recovery.subset == 'all_sampled') & (recovery.group == group) &
                               (recovery.truth_shape == truth) & (recovery.outcome == outcome)].iloc[0]
                axes[0].bar(position, row.percentage, bottom=bottom, color=colour, width=.8,
                            label=outcome.capitalize() if i == 0 and j == 0 else None)
                bottom += row.percentage
            np.testing.assert_allclose(bottom, 100.)
            coordinates.append(position)
            labels.append(f'{label}\n' + ('Gaussian truth' if j == 0 else 'Exponential truth'))
    axes[0].set(xticks=coordinates, xticklabels=labels, ylabel='Cells (%)', ylim=(0, 100))
    axes[0].tick_params(axis='x', rotation=25, labelsize=9)
    axes[0].legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.15), ncol=3)
    coordinates, labels = [], []
    for i, (group, label) in enumerate(zip(GROUPS, LABELS)):
        contrast = f'gaussian_minus_{matched_shape(group)}'
        for j, window in enumerate(['current', 'wider']):
            row = summary[(summary.record_type == 'contrast') & (summary.subset == 'all_sessions') &
                          (summary.level == 'group') & (summary.group == group) &
                          (summary.contrast == contrast) & (summary.window == window)].iloc[0]
            bottom, position = 0., 3 * i + j
            for column, name, colour in zip(['pct_favour_a', 'pct_favour_b', 'pct_undecided'],
                    ['Gaussian better', 'Exponential better', 'No clear difference'], palette):
                axes[1].bar(position, row[column], bottom=bottom, color=colour, width=.8,
                            label=name if i == 0 and j == 0 else None)
                bottom += row[column]
            np.testing.assert_allclose(bottom, 100.)
            axes[1].text(position, 102, p_label(row.wilcoxon_p), ha='center', va='bottom', fontsize=8)
            coordinates.append(position)
            labels.append(f'{label}\n{window.capitalize()}')
    axes[1].set(xticks=coordinates, xticklabels=labels, ylabel='Cells (%)', ylim=(0, 114), yticks=np.arange(0, 101, 20))
    axes[1].tick_params(axis='x', rotation=25, labelsize=9)
    axes[1].legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.15), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_undecided.png', dpi=200)
    plt.close(fig)


#%% run


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    (ROOT / 'data' / 'binned_wider').mkdir(parents=True, exist_ok=True)
    sample_cells()
    workers = min(12, os.cpu_count() or 1, len(SESSIONS_4S))
    print('session workers:', workers, flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run_both, SESSIONS_4S))
    recovery_summary()
    wider_summary()
    plot_checks()


if __name__ == '__main__':
    main()
