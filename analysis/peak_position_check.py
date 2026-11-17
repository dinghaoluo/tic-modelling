# -*- coding: utf-8 -*-
'''
Created on Thu Nov  5 19:15:36 2026

sort Gaussian peaks on one trial half and compare held-out shapes on the other

@author: Dinghao Luo
'''

#%% imports
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rec_list import SESSIONS_4S
from utils.rf_models import fit_field
from utils.trial_error_scores import trial_mse
from analysis.fit_shapes import cv_prediction
from analysis.compare_models_trial import trial_contrast, trial_loglikelihood
from analysis.compare_models import EXCLUDED, paired_test

#%% constants
OUTPUT  = ROOT / 'outputs' / 'peak_position'
FIGURES = ROOT / 'figures' / 'peak_position'
FITS    = ROOT / 'outputs' / 'fits'
WIDER   = ROOT / 'outputs' / 'wider_window_fits'
GROUPS  = ['tic_down', 'tic_up', 'time_cell']
LABELS  = ['Ramp-down TIC', 'Ramp-up TIC', 'Time cell']
CLASSES = ['start', 'inside', 'end']
CONTRASTS = [('gaussian', 'decay'), ('gaussian', 'rise'), ('decay', 'rise'), ('poly4', 'poly1')]
PLANNED = [('time_cell', 'start', 'gaussian_minus_decay'),
           ('time_cell', 'inside', 'gaussian_minus_decay'),
           ('tic_up', 'end', 'gaussian_minus_rise'),
           ('tic_up', 'inside', 'gaussian_minus_rise'),
           ('tic_down', 'inside', 'gaussian_minus_decay')]

#%% independent halves

def trial_halves(trial_rows):
    '''odd and even retain original one-based row parity, not retained-row position'''
    odd = (np.asarray(trial_rows) + 1) % 2 == 1
    even = ~odd
    assert not np.any(odd & even) and np.all(odd | even)
    return odd, even


def half_bootstrap(difference, testing, resamples):
    '''condition each saved paired trial resample on the half being evaluated'''
    keep = testing[resamples]
    counts = keep.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError('saved bootstrap draw has no opposite-half trials')
    draws = np.where(keep, difference[resamples], 0.).sum(axis=1) / counts
    return float(difference[testing].mean()), draws


def peak_class(mu, amplitude, end):
    peak = float(np.clip(mu, 0., end)) if amplitude > 0 else 0.
    label = 'start' if peak <= .1 * end else 'end' if peak >= .9 * end else 'inside'
    return peak, label


def session_analysis(session):
    started = time.perf_counter()
    fits = pd.read_csv(FITS / 'fits_all.csv')
    selected = fits[(fits.session == session) & fits.group.isin(GROUPS)].copy()
    folds = np.load(FITS / f'{session}_folds.npz')
    resamples = folds['bootstrap_trials']
    rows, sorts, failures, polynomial_rows = [], [], [], []
    for window, folder, source in [('current', 'binned', FITS / f'{session}_oof.h5'),
                                    ('wider', 'binned_wider', WIDER / f'{session}_wider_oof.h5')]:
        data = np.load(ROOT / 'data' / folder / f'{session}.npz')
        times, intervals = data['t'], data['T_k']
        np.testing.assert_array_equal(data['trial_rows'], folds['trial_rows'])
        odd, even = trial_halves(data['trial_rows'])
        end = float(np.median(intervals))
        sorting_bins = (times >= 0) & (times <= end)
        ids = data['neu_id'] if window == 'wider' else np.arange(1, len(data['dff']) + 1)
        axis = {int(neuron): i for i, neuron in enumerate(ids)}
        polynomial_file = None
        if window == 'wider':
            polynomial_file = h5py.File(OUTPUT / f'{session}_wider_poly_oof.h5', 'x')
            polynomial_file.create_dataset('neu_id', data=ids)
            polynomial_file.create_dataset('trial_rows', data=data['trial_rows'] + 1)
            polynomial_file.create_dataset('t', data=times)
            for model in ['poly1', 'poly4']:
                polynomial_file.create_dataset(model, shape=data['dff'].shape, dtype='f8', compression='lzf',
                                                chunks=(1, len(intervals), len(times)), fillvalue=np.nan)
        with h5py.File(source, 'r') as saved:
            np.testing.assert_array_equal(saved['trial_rows'][:], data['trial_rows'] + 1)
            np.testing.assert_allclose(saved['t'][:], times, rtol=0, atol=0)
            if window == 'wider':
                np.testing.assert_array_equal(saved['neu_id'][:], ids)
            for cell in selected.itertuples():
                i = axis[int(cell.neu_id)]
                signal = data['dff'][i].astype(float)
                predictions = {model: saved[model][i] for model in ['flat', 'gaussian', 'decay', 'rise']}
                for model in ['poly1', 'poly4']:
                    if window == 'current':
                        predictions[model] = saved[model][i]
                    else:
                        predictions[model], failed = cv_prediction(model, signal, times, intervals, folds['folds'])
                        polynomial_file[model][i] = predictions[model]
                        failures.extend([dict(window=window, neu_id=int(cell.neu_id), **f) for f in failed])
                for prediction in predictions.values():
                    np.testing.assert_array_equal(np.isfinite(prediction), np.isfinite(signal))
                variance = float(np.var((signal - predictions['flat'])[np.isfinite(signal)], ddof=0))
                if not np.isfinite(variance) or variance <= 0:
                    raise ValueError('invalid saved flat residual variance')
                errors = {model: trial_mse(signal, prediction) for model, prediction in predictions.items()}
                if window == 'wider':
                    difference = trial_contrast(errors['poly4'], errors['poly1'], variance)
                    lower, upper = np.percentile(difference[resamples].mean(axis=1), [2.5, 97.5])
                    polynomial_rows.append(dict(session=session, neu_id=int(cell.neu_id), group=cell.group,
                        mean_contrast=float(difference.mean()), lower=lower, upper=upper,
                        decision='poly4' if lower > 0 else 'poly1' if upper < 0 else 'undecided',
                        **{f'{m}_trial_mean_ll': float(trial_loglikelihood(errors[m], variance).mean())
                           for m in ['poly1', 'poly4']}))
                for direction, sorting, testing in [('odd', odd, even), ('even', even, odd)]:
                    assert not np.any(sorting & testing)
                    trial, bins = np.where(np.isfinite(signal) & sorting[:, None] & sorting_bins[None, :])
                    params, ok = fit_field('gaussian', times[bins], intervals[trial], signal[trial, bins])
                    peak, label = peak_class(params[2], params[1], end)
                    common = dict(window=window, session=session, neu_id=int(cell.neu_id), group=cell.group,
                                  sorting_half=direction, peak_class=label)
                    sort_row = dict(**common, gaussian_a0=params[0], gaussian_a=params[1],
                                    gaussian_mu=params[2], gaussian_sigma=params[3], peak_in_window=peak,
                                    sorting_end=end, mu_at_lower_bound=bool(np.isclose(params[2], -1., atol=1e-5, rtol=0)),
                                    negative_peak_with_rising_side=bool(-1. < params[2] < 0 and params[1] > 0 and not np.isclose(params[2], -1., atol=1e-5, rtol=0)),
                                    sorting_trials=int(sorting.sum()), testing_trials=int(testing.sum()),
                                    fit_success=bool(ok))
                    sorts.append(sort_row)
                    if not ok:
                        failures.append(dict(window=window, neu_id=int(cell.neu_id), model='gaussian',
                                             stage='sorting', sorting_half=direction))
                    for a, b in CONTRASTS:
                        difference = trial_contrast(errors[a], errors[b], variance)
                        mean, draws = half_bootstrap(difference, testing, resamples)
                        lower, upper = np.percentile(draws, [2.5, 97.5])
                        rows.append(dict(**sort_row, contrast=f'{a}_minus_{b}', model_a=a, model_b=b,
                                         mean_contrast=mean, lower=lower, upper=upper,
                                         decision=a if lower > 0 else b if upper < 0 else 'undecided',
                                         flat_residual_variance=variance, _draws=draws))
        if polynomial_file is not None:
            polynomial_file.close()
        data.close()
    # a cell can occupy different classes in the two directions; never duplicate it within a class
    directional = pd.DataFrame(rows)
    combined = []
    keys = ['window', 'session', 'neu_id', 'group', 'peak_class', 'contrast', 'model_a', 'model_b']
    for key, part in directional.groupby(keys, sort=False):
        draws = np.stack(part._draws).mean(axis=0)
        lower, upper = np.percentile(draws, [2.5, 97.5])
        a, b = key[-2:]
        combined.append(dict(zip(keys, key), sorting_half='combined', n_directions=len(part),
                             mean_contrast=float(part.mean_contrast.mean()), lower=lower, upper=upper,
                             decision=a if lower > 0 else b if upper < 0 else 'undecided'))
    directional = directional.drop(columns='_draws')
    combined = pd.concat([directional, pd.DataFrame(combined)], ignore_index=True)
    combined.to_csv(OUTPUT / f'{session}_cells.csv', index=False, mode='x')
    pd.DataFrame(sorts).to_csv(OUTPUT / f'{session}_sorting.csv', index=False, mode='x')
    pd.DataFrame(polynomial_rows).to_csv(OUTPUT / f'{session}_wider_polynomial_cells.csv', index=False, mode='x')
    status = dict(session=session, seconds=time.perf_counter() - started, failures=failures)
    (OUTPUT / f'{session}_status.json').write_text(json.dumps(status, indent=2))
    print(session, 'complete', flush=True)
    return status

#%% session inference
def holm(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.minimum(1., np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order]))
    result = np.empty_like(values)
    result[order] = adjusted
    return result


def summarise(cells, sorting):
    rows = []
    for window in ['current', 'wider']:
        for subset, excluded in [('all_sessions', []), ('excluded_sessions', EXCLUDED)]:
            frame = cells[(cells.window == window) & ~cells.session.isin(excluded)]
            combined = frame[frame.sorting_half == 'combined']
            direct = frame[frame.sorting_half != 'combined']
            for group in GROUPS:
                for label in CLASSES:
                    for a, b in CONTRASTS:
                        contrast = f'{a}_minus_{b}'
                        part = combined[(combined.group == group) & (combined.peak_class == label) &
                                        (combined.contrast == contrast)]
                        direction = direct[(direct.group == group) & (direct.peak_class == label) &
                                           (direct.contrast == contrast)]
                        medians = direction.groupby(['session', 'sorting_half']).mean_contrast.median().unstack()
                        counts = part.groupby('session').size()
                        eligible = medians.dropna().index.intersection(counts[counts >= 3].index)
                        values = medians.loc[eligible].mean(axis=1)
                        stat, p = paired_test(values) if len(values) >= 5 else (np.nan, np.nan)
                        common = dict(record_type='contrast', window=window, subset=subset, group=group,
                                      peak_class=label, contrast=contrast, model_a=a, model_b=b,
                                      planned=(group, label, contrast) in PLANNED)
                        rows.append(dict(**common, level='group', n_cells=len(part),
                            n_eligible_cells=int(part.session.isin(eligible).sum()), n_sessions=len(values),
                            n_sessions_favour_a=int((values > 0).sum()), n_sessions_favour_b=int((values < 0).sum()),
                            median_contrast=values.median(), wilcoxon_statistic=stat, p_raw=p,
                            pct_favour_a=100 * (part.decision == a).mean(),
                            pct_favour_b=100 * (part.decision == b).mean(),
                            pct_undecided=100 * (part.decision == 'undecided').mean()))
                        for session, value in values.items():
                            rows.append(dict(**common, level='session', session=session,
                                n_cells=int(counts[session]), n_sessions=1, median_contrast=value,
                                median_sort_odd=medians.loc[session, 'odd'], median_sort_even=medians.loc[session, 'even']))
            sort = sorting[(sorting.window == window) & ~sorting.session.isin(excluded)]
            for group in GROUPS:
                part = sort[sort.group == group]
                paired = part.pivot(index=['session', 'neu_id'], columns='sorting_half', values='peak_class')
                rows.append(dict(record_type='agreement', window=window, subset=subset, group=group,
                                 level='group', n_cells=len(paired), n_sessions=part.session.nunique(),
                                 pct_same_class=100 * (paired.odd == paired.even).mean()))
                for direction in ['odd', 'even']:
                    for label in CLASSES:
                        selected = part[(part.sorting_half == direction) & (part.peak_class == label)]
                        rows.append(dict(record_type='class_count', window=window, subset=subset, group=group,
                            sorting_half=direction, peak_class=label, level='group', n_cells=len(selected),
                            n_sessions=selected.session.nunique()))
            start = sort[(sort.group == 'time_cell') & (sort.peak_class == 'start')]
            for bound, selected in [('all', start), ('at_lower_bound', start[start.mu_at_lower_bound]),
                                     ('not_at_lower_bound', start[~start.mu_at_lower_bound])]:
                rows.append(dict(record_type='start_peak', window=window, subset=subset, group='time_cell',
                    peak_class='start', level='group', bound_subset=bound, n_cells=selected.neu_id.groupby(selected.session).nunique().sum(),
                    n_sorting_fits=len(selected), median_fitted_peak=selected.gaussian_mu.median(),
                    fitted_peak_q25=selected.gaussian_mu.quantile(.25), fitted_peak_q75=selected.gaussian_mu.quantile(.75),
                    pct_peak_before_zero=100 * (selected.gaussian_mu < 0).mean(),
                    pct_negative_peak_with_rising_side=100 * selected.negative_peak_with_rising_side.mean(),
                    pct_at_lower_bound=100 * selected.mu_at_lower_bound.mean()))
    result = pd.DataFrame(rows)
    result['p_holm'] = np.nan
    for window in ['current', 'wider']:
        for subset in ['all_sessions', 'excluded_sessions']:
            index = result.index[(result.record_type == 'contrast') & (result.level == 'group') &
                                 (result.window == window) & (result.subset == subset) & result.planned.fillna(False)]
            # unavailable planned tests remain in the family, conservatively represented by one
            corrected = holm(result.loc[index, 'p_raw'].fillna(1.).to_numpy())
            result.loc[index, 'p_holm'] = np.where(result.loc[index, 'p_raw'].notna(), corrected, np.nan)
    return result


def full_data_start_peaks(sorting):
    rows, cell_rows = [], []
    for window in ['current', 'wider']:
        parameters = (pd.read_csv(FITS / 'fits_all.csv') if window == 'current' else
                      pd.concat([pd.read_csv(WIDER / f'{session}_wider_parameters.csv')
                                 for session in SESSIONS_4S], ignore_index=True))
        for subset, excluded in [('all_sessions', []), ('excluded_sessions', EXCLUDED)]:
            selected = sorting[(sorting.window == window) & (sorting.group == 'time_cell') &
                               (sorting.peak_class == 'start') & ~sorting.session.isin(excluded)]
            for direction in ['odd', 'even', 'either_half']:
                ids = selected if direction == 'either_half' else selected[selected.sorting_half == direction]
                ids = ids[['session', 'neu_id']].drop_duplicates()
                cells = ids.merge(parameters, on=['session', 'neu_id'], validate='one_to_one')
                at_bound = np.isclose(cells.gaussian_mu, -1., atol=1e-5, rtol=0)
                cells['at_lower_bound'] = at_bound
                cells['negative_peak_with_rising_side'] = ((cells.gaussian_mu > -1) & (cells.gaussian_mu < 0) &
                                                            (cells.gaussian_a > 0) & ~at_bound)
                for bound, part in [('all', cells), ('at_lower_bound', cells[at_bound]),
                                     ('not_at_lower_bound', cells[~at_bound])]:
                    rows.append(dict(record_type='full_data_start_peak', window=window, subset=subset,
                        group='time_cell', sorting_half=direction, bound_subset=bound, n_cells=len(part),
                        n_sessions=part.session.nunique(), median_fitted_peak=part.gaussian_mu.median(),
                        fitted_peak_q25=part.gaussian_mu.quantile(.25), fitted_peak_q75=part.gaussian_mu.quantile(.75),
                        pct_peak_before_zero=100 * (part.gaussian_mu < 0).mean(),
                        pct_negative_peak_with_rising_side=100 * part.negative_peak_with_rising_side.mean(),
                        pct_at_lower_bound=100 * part.at_lower_bound.mean()))
                cell_rows.extend([dict(window=window, subset=subset, sorting_half=direction,
                                       **row) for row in cells[['session', 'neu_id', 'gaussian_mu', 'gaussian_sigma',
                                       'at_lower_bound', 'negative_peak_with_rising_side']].to_dict('records')])
    return pd.DataFrame(rows), pd.DataFrame(cell_rows)


#%% figure and optimisation trace
def p_text(value):
    if not np.isfinite(value):
        return 'not tested'
    return 'p < 0.001' if value < .001 else f'p = {value:#.2g}' + (' (n.s.)' if value >= .05 else '')


def make_figure(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12})
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=True)
    annotations = []
    for row, window in enumerate(['current', 'wider']):
        for column, (group, label) in enumerate(zip(GROUPS, LABELS)):
            ax = axes[row, column]
            ticks, ticklabels = [], []
            for i, peak in enumerate(CLASSES):
                exponential = 'decay' if peak == 'start' else 'rise' if peak == 'end' else 'rise' if group == 'tic_up' else 'decay'
                selected = summary[(summary.record_type == 'contrast') & (summary.level == 'group') &
                                   (summary.subset == 'all_sessions') & (summary.window == window) &
                                   (summary.group == group) & (summary.peak_class == peak) &
                                   (summary.contrast == f'gaussian_minus_{exponential}')].iloc[0]
                if selected.n_sessions < 5:
                    continue
                values = selected[['pct_favour_a', 'pct_favour_b', 'pct_undecided']].to_numpy(float)
                np.testing.assert_allclose(values.sum(), 100., atol=1e-10)
                bottom = 0.
                for value, colour in zip(values, ['#4477AA', '#EE6677', '#BBBBBB']):
                    ax.bar(i, value, bottom=bottom, color=colour, width=.65)
                    bottom += value
                p = selected.p_holm if selected.planned else selected.p_raw
                text = p_text(p)
                ax.text(i, 103, text, ha='center', va='bottom', fontsize=10)
                ticks.append(i)
                ticklabels.append(f'Peak {"at " if peak != "inside" else ""}{peak}\nn = {int(selected.n_cells)}')
                annotations.append(dict(window=window, group=group, peak_class=peak, contrast=selected.contrast,
                                        p=float(p), text=text, n_cells=int(selected.n_cells)))
            ax.set(xticks=ticks, xticklabels=ticklabels, ylim=(0, 114), yticks=[0, 25, 50, 75, 100], title=label if row == 0 else '')
            if column == 0:
                ax.set_ylabel(('Current window' if row == 0 else 'Wider window') + '\nCells (%)')
    fig.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(
        ['#4477AA', '#EE6677', '#BBBBBB'], ['Gaussian better', 'Exponential better', 'No clear difference'])],
        loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.5, .99))
    fig.subplots_adjust(top=.89, bottom=.08, left=.07, right=.98, hspace=.35, wspace=.16)
    fig.savefig(FIGURES / 'fig_peak_position.png', dpi=200)
    plt.close(fig)
    return annotations


def optimisation_trace(cells):
    flags = json.loads((WIDER / 'verification.json').read_text())['failures']
    rows = []
    for flag in flags:
        group_rows = cells[(cells.session == flag['session']) & (cells.neu_id == flag['neu_id'])]
        group = group_rows.group.iloc[0]
        memberships = group_rows[(group_rows.window == 'wider') & (group_rows.sorting_half != 'combined')]
        touched = []
        if flag['test'] == 'wider':
            for planned_group, label, contrast in PLANNED:
                if planned_group != group or flag['model'] not in contrast.split('_minus_'):
                    continue
                part = memberships[(memberships.peak_class == label) & (memberships.contrast == contrast)]
                if len(part):
                    touched.extend([dict(group=group, peak_class=label, contrast=contrast, sorting_half=x)
                                    for x in part.sorting_half.tolist()])
        rows.append(dict(**flag, group=group, used_in_peak_scores=flag['test'] == 'wider',
                         planned_memberships=touched))
    return rows


#%% main
def main():
    started = time.perf_counter()
    for directory in [OUTPUT, FIGURES]:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f'new result directory must be empty: {directory}')
        directory.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(OUTPUT / 'matplotlib')
    protected = {str(path): (path.stat().st_size, path.stat().st_mtime_ns)
                 for folder in ['outputs', 'figures'] for path in (ROOT / folder).rglob('*')
                 if path.is_file() and OUTPUT not in path.parents and FIGURES not in path.parents}
    workers = min(12, os.cpu_count() or 1, len(SESSIONS_4S))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(session_analysis, session) for session in SESSIONS_4S]
        status = [future.result() for future in pending]
    cells = pd.concat([pd.read_csv(OUTPUT / f'{session}_cells.csv') for session in SESSIONS_4S], ignore_index=True)
    sorting = pd.concat([pd.read_csv(OUTPUT / f'{session}_sorting.csv') for session in SESSIONS_4S], ignore_index=True)
    summary = summarise(cells, sorting)
    start_summary, start_cells = full_data_start_peaks(sorting)
    start_summary.to_csv(OUTPUT / 'peak_position_start_peak_summary.csv', index=False, mode='x')
    start_cells.to_csv(OUTPUT / 'peak_position_start_peak_cells.csv', index=False, mode='x')
    cells.to_csv(OUTPUT / 'peak_position_cells.csv', index=False, mode='x')
    sorting.to_csv(OUTPUT / 'peak_position_sorting.csv', index=False, mode='x')
    summary.to_csv(OUTPUT / 'peak_position_summary.csv', index=False, mode='x')
    annotations = make_figure(summary)
    trace = optimisation_trace(cells)
    (OUTPUT / 'optimisation_trace.json').write_text(json.dumps(trace, indent=2))
    for path, metadata in protected.items():
        stat = Path(path).stat()
        assert (stat.st_size, stat.st_mtime_ns) == metadata, path
    report = dict(seconds=time.perf_counter() - started, workers=workers, interpreter=sys.executable,
                  old_files_unchanged=True, failures=[dict(session=s['session'], **f) for s in status for f in s['failures']],
                  figure_annotations=annotations,
                  sorting='Gaussian fit to observed bins with centres from zero through session median T; continuous in-window peak; first maximiser for a zero-amplitude fit',
                  scoring='opposite original-row parity only; saved out-of-fold predictions may have been trained on sorting-half trials',
                  bootstrap='saved paired resamples conditioned on testing half; two direction estimates averaged when a cell has the same class',
                  session_statistic='average of odd-sort and even-sort class session medians; at least three distinct cells and both directions present',
                  correction='Holm across the five planned tests separately for each window and session subset; unavailable tests count in the family',
                  start_peak='unconstrained fitted mu; rising side in wider interval means -1 < mu < 0 and positive amplitude; lower-bound fits reported separately')
    (OUTPUT / 'run_checks.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
