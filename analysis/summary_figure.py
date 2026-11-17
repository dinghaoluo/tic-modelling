# -*- coding: utf-8 -*-
'''
Created on Thu Nov  5 18:05:57 2026

summarise held-out shapes and wider-window field parameters

@author: Dinghao Luo
'''

#%% imports
import argparse
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
from utils.trial_error_scores import trial_mse
from analysis.compare_models_trial import trial_loglikelihood, trial_contrast
from analysis.compare_models import paired_test
from utils.rf_models_v1_2 import BOUNDS

#%% constants
OUTPUT = ROOT / 'outputs' / 'model_comparison'
FIGURES = ROOT / 'figures'
FITS = ROOT / 'outputs' / 'fits_v1_2'
GROUPS = ['tic_down', 'tic_up', 'time_cell']
LABELS = ['Ramp-down TIC', 'Ramp-up TIC', 'Time cells']
COLOURS = ['#228833', '#4477AA', '#9467BD']
PRIMARY = {'tic_down': 'gaussian_minus_decay', 'tic_up': 'gaussian_minus_rise',
           'time_cell': 'gaussian_minus_decay'}
BASE_COLUMNS = ['group', 'measure', 'n cells', 'n sessions', 'median', 'IQR low', 'IQR high',
                'SD', '% cells at bound', 'comparison', 'p for median', 'p for SD']

#%% polynomial predictions
def polynomial_session(session):
    started = time.perf_counter()
    fits = pd.read_csv(FITS / 'fits_all.csv')
    fits = fits[fits.session == session].reset_index(drop=True)
    data = np.load(ROOT / 'data' / 'binned_v1_1' / f'{session}.npz')
    dff = data['dff']
    folds = np.load(FITS / f'{session}_folds.npz')
    np.testing.assert_array_equal(fits.neu_id, np.arange(1, len(dff) + 1))
    np.testing.assert_array_equal(data['trial_rows'], folds['trial_rows'])
    resamples = folds['bootstrap_trials']
    if resamples.shape[1] != dff.shape[1] or np.any(resamples < 0) or np.any(resamples >= dff.shape[1]):
        raise ValueError(f'{session}: bootstrap trials do not match observations')
    rows = []
    with h5py.File(FITS / f'{session}_oof.h5', 'r') as saved:
        np.testing.assert_array_equal(saved['trial_rows'][:], data['trial_rows'] + 1)
        for neuron, cell in fits.iterrows():
            if cell.group not in GROUPS:
                continue
            signal = dff[neuron]
            flat = saved['flat'][neuron]
            np.testing.assert_array_equal(np.isfinite(flat), np.isfinite(signal))
            flat_error = trial_mse(signal, flat)
            np.testing.assert_allclose(flat_error.mean(), cell.flat_held_mse, rtol=0, atol=1e-8)
            variance = np.var((signal - flat)[np.isfinite(signal)], ddof=0)
            if not np.isfinite(variance) or variance <= 0:
                raise ValueError(f'{session}, neuron {neuron + 1}: invalid flat variance')
            errors = {}
            row = dict(session=session, neu_id=int(cell.neu_id), group=cell.group,
                       flat_residual_variance=variance)
            for degree in range(1, 5):
                model = f'poly{degree}'
                prediction = saved[model][neuron]
                np.testing.assert_array_equal(np.isfinite(prediction), np.isfinite(signal))
                errors[degree] = trial_mse(signal, prediction)
                np.testing.assert_allclose(errors[degree].mean(), cell[f'{model}_held_mse'], rtol=0, atol=1e-8)
                row[f'{model}_trial_mean_ll'] = trial_loglikelihood(errors[degree], variance).mean()
            difference = trial_contrast(errors[4], errors[1], variance)
            lower, upper = np.percentile(difference[resamples].mean(axis=1), [2.5, 97.5])
            row.update(mean_contrast=difference.mean(), lower=lower, upper=upper,
                       decision='poly4' if lower > 0 else 'poly1' if upper < 0 else 'undecided')
            rows.append(row)
    print(session, 'polynomial check complete in seconds', time.perf_counter() - started, flush=True)
    return rows


#%% table statistics
def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('missing or non-finite summary distribution')
    low, median, high = np.percentile(values, [25, 50, 75])
    return {'median': median, 'IQR low': low, 'IQR high': high,
            'SD': np.std(values, ddof=1) if len(values) > 1 else np.nan}


def p_label(p):
    if not np.isfinite(p):
        raise ValueError('figure p-value is undefined')
    if p < .001:
        return 'p < 0.001'
    return f'p = {p:#.2g}' + (' (n.s.)' if p >= .05 else '')


def category_summary(cells, a, b):
    medians = cells.groupby('session').mean_contrast.median()
    _, p = paired_test(medians)
    percentages = [100 * (cells.decision == name).mean() for name in [a, b, 'undecided']]
    np.testing.assert_allclose(sum(percentages), 100., atol=1e-10)
    return percentages, p, len(medians)


def parameter_name(group, measure):
    if measure == 'peak':
        return 'gaussian_mu' if group == 'time_cell' else ('rise_mu' if group == 'tic_up' else 'decay_mu')
    if measure == 'width':
        return 'gaussian_sigma'
    return 'rise_tau' if group == 'tic_up' else 'decay_tau'


def bound_percentages(cells, parameter):
    bound_parameter = 'rise_mu' if parameter == 'peak_reference' else parameter
    values = cells.loc[cells[f'{bound_parameter}_at_bound'].astype(bool), parameter].dropna().to_numpy()
    if parameter.startswith('rise_') and len(values) and np.nanmedian(values) < 0:
        values = -values
    if not len(values):
        return 0., 0.
    model = 'gaussian' if parameter.startswith('gaussian_') else 'rise' if parameter.startswith('rise_') else 'decay'
    limit_index = 0 if parameter.endswith('_mu') else 1
    lower_limit, upper_limit = BOUNDS[model][0][limit_index], BOUNDS[model][1][limit_index]
    lower = np.isclose(values, lower_limit, rtol=0., atol=1e-5)
    upper = np.isclose(values, upper_limit, rtol=0., atol=1e-5)
    return 100. * lower.sum() / len(cells), 100. * upper.sum() / len(cells)


def parameter_values(fits, group, measure, include_bound=False):
    parameter = parameter_name(group, measure)
    cells = fits[fits.group == group]
    cells = cells if include_bound else cells[~cells[f'{parameter}_at_bound'].astype(bool)]
    cells = cells.loc[cells[parameter].notna()].copy()
    if measure == 'peak':
        if group == 'tic_up':
            intervals = {}
            for session in cells.session.drop_duplicates():
                data = np.load(ROOT / 'data' / 'binned_v1_3' / f'{session}.npz')
                intervals[session] = float(np.nanmedian(data['T_k']))
            cells['peak_reference'] = cells.apply(lambda row: intervals[row.session] - row[parameter], axis=1)
            parameter = 'peak_reference'
    bound_parameter = 'rise_mu' if group == 'tic_up' and measure == 'peak' else parameter
    return cells, parameter, 100 * fits.loc[fits.group == group, f'{bound_parameter}_at_bound'].astype(bool).mean()


def paired_parameters(left, right, left_parameter, right_parameter):
    pairs = []
    for session in left.session.drop_duplicates():
        a = left.loc[left.session == session, left_parameter]
        b = right.loc[right.session == session, right_parameter]
        if not len(a) or not len(b):
            continue
        pairs.append(dict(session=session, n_group=len(a), n_time_cell=len(b),
                          median_group=a.median(), median_time_cell=b.median(),
                          sd_group=a.std(ddof=1), sd_time_cell=b.std(ddof=1)))
    if not pairs:
        raise ValueError('no eligible paired parameter sessions')
    pairs = pd.DataFrame(pairs)
    _, median_p = paired_test(pairs.median_group - pairs.median_time_cell)
    _, sd_p = paired_test(pairs.sd_group - pairs.sd_time_cell)
    return median_p, sd_p, pairs


def make_table(fits, primary, polynomial, population, literature):
    rows, pair_rows = [], []
    for group, label in zip(GROUPS, LABELS):
        selected = primary[(primary.group == group) & (primary.contrast == PRIMARY[group])]
        np.testing.assert_array_equal(selected.neu_id.to_numpy(), fits.loc[fits.group == group, 'neu_id'].to_numpy())
        exponential = 'rise' if group == 'tic_up' else 'decay'
        percentages, p, n_sessions = category_summary(selected, 'gaussian', exponential)
        reference = population[(population.subset == 'all_sessions') & (population.level == 'group') &
                               (population.group == group) & (population.contrast == PRIMARY[group])]
        if len(reference) != 1:
            raise ValueError('primary population row missing or duplicated')
        reference = reference.iloc[0]
        np.testing.assert_allclose(percentages, reference[['pct_favour_a', 'pct_favour_b', 'pct_undecided']].to_numpy(float), atol=1e-10)
        np.testing.assert_allclose(p, reference.wilcoxon_p, atol=1e-12)
        assert len(selected) == reference.n_cells and n_sessions == reference.n_sessions
        rows.append({'group': label, 'measure': 'Gaussian vs exponential', 'n cells': len(selected),
                     'n sessions': n_sessions, **distribution(selected.mean_contrast),
                     'comparison': f'Gaussian minus {exponential}', 'p for median': p,
                     '% Gaussian better': percentages[0], '% Exponential better': percentages[1],
                     '% No clear difference': percentages[2]})
        selected = polynomial[polynomial.group == group]
        percentages, p, n_sessions = category_summary(selected, 'poly4', 'poly1')
        rows.append({'group': label, 'measure': 'Linear vs curved', 'n cells': len(selected),
                     'n sessions': n_sessions, **distribution(selected.mean_contrast),
                     'comparison': 'degree 4 minus degree 1', 'p for median': p,
                     '% Curved better': percentages[0], '% Linear better': percentages[1],
                     '% No clear difference': percentages[2]})
        for measure in ['peak', 'width', 'ramp rate']:
            retained, parameter, bound_percentage = parameter_values(fits, group, measure)
            row = {'group': label, 'measure': measure, 'n cells': len(retained),
                   'n sessions': retained.session.nunique(), **distribution(retained[parameter]),
                   '% cells at bound': bound_percentage,
                   'comparison': 'reference' if group == 'time_cell' else 'Time cell',
                   'source parameter': ('-rise_mu' if group == 'tic_up' and measure == 'peak' else parameter),
                   'convention': ('session median T_k minus rise_mu, relative to last lick' if group == 'tic_up' and measure == 'peak'
                                  else 'decay_mu, relative to last lick' if group == 'tic_down' and measure == 'peak'
                                  else '')}
            if group != 'time_cell':
                reference_cells, reference_parameter, _ = parameter_values(fits, 'time_cell', measure)
                p, sd_p, pairs = paired_parameters(retained, reference_cells, parameter, reference_parameter)
                if not (measure == 'peak' and group == 'tic_up'):
                    row.update({'n sessions': len(pairs), 'p for median': p, 'p for SD': sd_p})
                    pair_rows.extend([dict(group=label, measure=measure,
                                           convention=('session median T_k minus rise_mu, relative to last lick' if group == 'tic_up' and measure == 'peak'
                                                       else 'decay_mu, relative to last lick' if group == 'tic_down' and measure == 'peak' else ''),
                                           **record) for record in pairs.to_dict('records')])
            if measure == 'ramp rate' and group in ['tic_down', 'tic_up']:
                row.update({'published value': 2.25 if group == 'tic_down' else 1.03,
                            'citation': 'published ramp benchmark'})
            rows.append(row)
        for degree in range(1, 5):
            selected = polynomial[polynomial.group == group]
            rows.append({'group': label, 'measure': f'polynomial degree {degree}', 'n cells': len(selected),
                         'n sessions': selected.session.nunique(),
                         **distribution(selected[f'poly{degree}_trial_mean_ll']),
                         'comparison': 'mean held-out log-likelihood per observed bin, equal trial weights'})
    for metric, subset, measure in [('peak_power_law_alpha', None, 'time-cell exponent'),
                                    ('width_on_peak_slope_trial_check', 'field_only', 'field-only width slope')]:
        selected = literature[(literature.group == 'time_cell') & (literature.metric == metric)]
        if subset is not None:
            selected = selected[selected.subset == subset]
        if len(selected) != 1:
            raise ValueError(f'published comparison row missing: {measure}')
        value = selected.iloc[0]
        row = {'group': 'Time cells', 'measure': measure, 'n cells': value.n_cells,
               'n sessions': fits.loc[fits.group == 'time_cell', 'session'].nunique(),
               'estimate': value.our_value, 'standard error': value.our_standard_error,
               'interval low': value.our_lower, 'interval high': value.our_upper,
               'published value': value.published_value, 'published error': value.published_error,
               'published interval low': value.published_lower, 'published interval high': value.published_upper,
               'citation': value.citation}
        if measure == 'time-cell exponent':
            row['n cells in peak limits'] = value.n_peaks_in_limits
        else:
            row['n sessions'] = fits.loc[(fits.group == 'time_cell') & (fits.gaussian_class == 'field'), 'session'].nunique()
        rows.append(row)
    peak_cells = {}
    for group in GROUPS:
        peak_cells[group] = parameter_values(parameters, group, 'peak', include_bound=True)[0]
    cross_left = peak_cells['tic_down']
    cross_right = peak_cells['tic_up']
    cross_pairs = paired_parameters(cross_left, cross_right, parameter_name('tic_down', 'peak'), parameter_name('tic_up', 'peak'))[2]
    _, cross_p = paired_test(cross_pairs.median_group - cross_pairs.median_time_cell)
    rows.append({'group': 'Ramp-down TIC vs Ramp-up TIC', 'measure': 'peak', 'n cells': len(cross_left) + len(cross_right),
                 'n sessions': len(cross_pairs), 'median': np.nan, 'IQR low': np.nan, 'IQR high': np.nan,
                 'SD': np.nan, 'comparison': 'shared last-lick reference', 'p for median': cross_p,
                 'convention': 'ramp-down decay_mu; ramp-up session median T_k minus rise_mu'})
    for record in cross_pairs.to_dict('records'):
        pair_rows.append(dict(group='Ramp-down TIC vs Ramp-up TIC', measure='peak', bound_subset='included',
                              convention='ramp-down decay_mu; ramp-up session median T_k minus rise_mu', **record))
    table = pd.DataFrame(rows)
    for column in BASE_COLUMNS:
        if column not in table:
            table[column] = np.nan
    return table[BASE_COLUMNS + [column for column in table if column not in BASE_COLUMNS]], pd.DataFrame(pair_rows)


#%% figure
def make_figure(table, fits, suffix='', wider=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.size': 16, 'axes.titlesize': 19, 'axes.labelsize': 16,
                         'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 14,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'pdf.use14corefonts': False})
    fig = plt.figure(figsize=(13., 11.))
    grid = fig.add_gridspec(2, 6, height_ratios=[1., 1.15], hspace=.6, wspace=1.)
    axes = [fig.add_subplot(grid[0, :3]), fig.add_subplot(grid[0, 3:]),
            fig.add_subplot(grid[1, :2]), fig.add_subplot(grid[1, 2:4]), fig.add_subplot(grid[1, 4:])]
    annotations = []
    for ax, panel, title, categories in [
            (axes[0], 'A', 'Gaussian vs exponential', ['Gaussian better', 'Exponential better', 'No clear difference']),
            (axes[1], 'B', 'Linear vs curved', ['Curved better', 'Linear better', 'No clear difference'])]:
        selected = table[table.measure == title].set_index('group').loc[LABELS]
        bottom = np.zeros(len(GROUPS))
        for category, colour in zip(categories, ['#4D4D4D', '#E69F00', '#BDBDBD']):
            values = selected['% ' + category].to_numpy(float)
            ax.bar(np.arange(len(GROUPS)), values, bottom=bottom, color=colour, width=.65, label=category)
            bottom += values
        np.testing.assert_allclose(bottom, 100., atol=1e-10)
        for i, (label, row) in enumerate(selected.iterrows()):
            text = p_label(row['p for median'])
            ax.text(i, 103, text, ha='center', va='bottom', fontsize=15)
            annotations.append(dict(panel=panel, group=label, measure=title, p=float(row['p for median']), text=text))
        ax.set(title=title, ylabel='Cells (%)', ylim=(0, 114), yticks=np.arange(0, 101, 20),
               xticks=range(len(LABELS)), xticklabels=[f'{label}\n(n = {int(selected.loc[label, "n cells"])})' for label in LABELS])
        ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(.5, 1.14), ncol=2, fontsize=14,
                  handlelength=1., columnspacing=1.)
    for ax, panel, measure, title, ylabel in zip(axes[2:], ['C', 'D', 'E'],
            ['peak', 'width', 'ramp rate'], ['Peak location', 'Gaussian width', 'Time constant'],
            ['Peak location (s)', 'Gaussian width, \u03c3 (s)', 'Time constant, \u03c4 (s)']):
        values = [parameter_values(fits, group, measure, include_bound=wider)[0][parameter_values(fits, group, measure)[1]].to_numpy()
                  for group in GROUPS]
        boxes = ax.boxplot(values, tick_labels=LABELS, showfliers=False, patch_artist=True,
                           medianprops={'color': 'black', 'linewidth': 1.5})
        for box, colour in zip(boxes['boxes'], COLOURS):
            box.set(facecolor=colour, alpha=.55)
        if wider:
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
                row = table[(table.group == 'Ramp-down TIC vs Ramp-up TIC') & (table.measure == measure)].iloc[0]
            height = high + (.08 + .19 * i) * span
            ax.plot([left + 1, left + 1, right + 1, right + 1], [height, height + .035 * span, height + .035 * span, height],
                    color='black', linewidth=.8)
            text = p_label(row['p for median'])
            ax.text((left + right + 2) / 2, height + .05 * span, text, ha='center', va='bottom', fontsize=15)
            annotations.append(dict(panel=panel, group=str(row['group']), measure=measure, p=float(row['p for median']), text=text))
        if panel in ['C', 'E']:
            bracket_top = .85 if panel == 'C' else 1.0
            ax.set_ylim(low, high + bracket_top * span)
        else:
            ax.set_ylim(low, high + .6 * span)
        if measure == 'width' and wider:
            ax.legend(handles=[Line2D([0], [0], color='black', marker='x', linestyle='none',
                       label='At bound')], frameon=False, loc='upper left', fontsize=14)
        if measure == 'ramp rate':
            ax.hlines(2.25, .78, 1.22, colors='black', linewidth=2.0, linestyles='-')
            ax.hlines(1.03, 1.78, 2.22, colors='black', linewidth=2.0, linestyles='--')
            ax.legend(handles=[Line2D([0], [0], color='black', linewidth=2.0, linestyle='-', label='Cao et al. 2024 past'),
                               Line2D([0], [0], color='black', linewidth=2.0, linestyle='--', label='Cao et al. 2024 future')],
                      frameon=False, loc='upper left', fontsize=14)
    for ax, letter in zip(axes, 'ABCDE'):
        ax.text(-.12, 1.05, letter, transform=ax.transAxes, fontsize=22, fontweight='bold', ha='right')
    fig.subplots_adjust(left=.085, right=.98, bottom=.12, top=.84)
    name = 'receptive_field_model_summary' if wider else f'fig_summary{suffix}'
    fig.savefig(FIGURES / f'{name}.png', dpi=300)
    fig.savefig(FIGURES / f'{name}.pdf')
    plt.close(fig)
    return annotations


#%% main
def main():
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fits = pd.read_csv(FITS / 'fits_all.csv')
    primary = pd.read_csv(OUTPUT / 'cell_ll_trial.csv')
    population = pd.read_csv(OUTPUT / 'population_ll_trial.csv')
    literature = pd.read_csv(OUTPUT / 'literature.csv')
    sessions = fits.session.drop_duplicates().tolist()
    workers = min(12, os.cpu_count() or 1, len(sessions))
    print('session workers:', workers, flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(polynomial_session, sessions))
    polynomial = pd.DataFrame([row for session in results for row in session])
    polynomial.to_csv(OUTPUT / 'summary_polynomial_cells.csv', index=False)
    table, pairs = make_table(fits, primary, polynomial, population, literature)
    table.to_csv(OUTPUT / 'summary_table.csv', index=False)
    pairs.to_csv(OUTPUT / 'summary_parameter_session_pairs.csv', index=False)
    annotations = make_figure(table, fits)
    saved_table = pd.read_csv(OUTPUT / 'summary_table.csv')
    for annotation in annotations:
        row = saved_table[(saved_table.group == annotation['group']) & (saved_table.measure == annotation['measure'])]
        assert len(row) == 1
        np.testing.assert_allclose(row.iloc[0]['p for median'], annotation['p'], rtol=0, atol=1e-14)
        assert p_label(row.iloc[0]['p for median']) == annotation['text']
    report = dict(seconds=time.perf_counter() - started, workers=workers, figure_p_values=annotations,
                  bar_percentages_sum_to_100=True, primary_matches_population=True,
                  every_figure_p_matches_table=True, interpreter=sys.executable,
                  bound_exclusion='parameter-specific bound flags only; percentages use all group cells as denominator',
                  parameter_tests='two-sided paired session median and SD tests; at least five retained cells in both groups',
                  n_sessions='paired-session count for TIC parameter comparisons; group session count otherwise',
                  polynomial_likelihood='one flat residual population variance per cell; equal trial weights; saved trial resamples; all four degrees from saved predictions')
    (OUTPUT / 'summary_figure_checks.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)



def wider_main():
    started = time.perf_counter()
    wider_dir = ROOT / 'outputs' / 'undecided_check'
    peak_dir = ROOT / 'outputs' / 'peak_position'
    os.environ['MPLCONFIGDIR'] = str(peak_dir / 'matplotlib')
    protected = {str(path): (path.stat().st_size, path.stat().st_mtime_ns)
                 for folder in ['outputs', 'figures'] for path in (ROOT / folder).rglob('*') if path.is_file()}
    parameters = pd.concat([pd.read_csv(wider_dir / f'{session}_wider_parameters.csv')
                            for session in pd.read_csv(FITS / 'fits_all.csv').session.drop_duplicates()], ignore_index=True)
    primary = pd.read_csv(wider_dir / 'wider_window_cells.csv')
    polynomial = pd.concat([pd.read_csv(peak_dir / f'{session}_wider_polynomial_cells.csv')
                             for session in parameters.session.drop_duplicates()], ignore_index=True)
    rows, pair_rows = [], []
    for group, label in zip(GROUPS, LABELS):
        for measure, cells, a, b, categories in [
                ('Gaussian vs exponential', primary[(primary.group == group) & (primary.contrast == PRIMARY[group])],
                 'gaussian', 'rise' if group == 'tic_up' else 'decay', ['Gaussian better', 'Exponential better', 'No clear difference']),
                ('Linear vs curved', polynomial[polynomial.group == group], 'poly4', 'poly1',
                 ['Curved better', 'Linear better', 'No clear difference'])]:
            medians = cells.groupby('session').mean_contrast.agg(['size', 'median'])
            eligible = medians.dropna()
            _, p = paired_test(eligible['median'])
            percentages = [100 * (cells.decision == name).mean() for name in [a, b, 'undecided']]
            np.testing.assert_allclose(sum(percentages), 100., atol=1e-10)
            rows.append({'group': label, 'measure': measure, 'n cells': len(cells), 'n sessions': len(eligible),
                         **distribution(cells.mean_contrast), 'comparison': f'{a} minus {b}', 'p for median': p,
                         **{'% ' + category: value for category, value in zip(categories, percentages)}})
        for measure in ['peak', 'width', 'ramp rate']:
            all_left, parameter, _ = parameter_values(parameters, group, measure, include_bound=True)
            all_right, right_parameter, _ = parameter_values(parameters, 'time_cell', measure, include_bound=True)
            unbound_left, _, _ = parameter_values(parameters, group, measure, include_bound=False)
            unbound_right, _, _ = parameter_values(parameters, 'time_cell', measure, include_bound=False)
            lower_pct, upper_pct = bound_percentages(all_left, parameter)
            if group != 'time_cell':
                p_in, sd_in, pairs_in = paired_parameters(all_left, all_right, parameter, right_parameter)
                p_ex, sd_ex, pairs_ex = paired_parameters(unbound_left, unbound_right, parameter, right_parameter)
                if not (measure == 'peak' and group == 'tic_up'):
                    pair_rows.extend([dict(group=label, measure=measure, bound_subset='included',
                                           convention=('session median T_k minus rise_mu, relative to last lick' if group == 'tic_up' and measure == 'peak'
                                                       else 'decay_mu, relative to last lick' if group == 'tic_down' and measure == 'peak' else ''),
                                           **record) for record in pairs_in.to_dict('records')])
                    pair_rows.extend([dict(group=label, measure=measure, bound_subset='excluded',
                                           convention=('session median T_k minus rise_mu, relative to last lick' if group == 'tic_up' and measure == 'peak'
                                                       else 'decay_mu, relative to last lick' if group == 'tic_down' and measure == 'peak' else ''),
                                           **record) for record in pairs_ex.to_dict('records')])
            else:
                p_in = sd_in = p_ex = sd_ex = np.nan
            for bound_subset, retained, p_value, sd_value in [
                    ('included', all_left, p_in, sd_in),
                    ('excluded', unbound_left, p_ex, sd_ex)]:
                row = {'group': label, 'measure': measure, 'bound subset': bound_subset,
                       'n cells': len(retained), 'n sessions': retained.session.nunique(),
                       **distribution(retained[parameter]), '% lower bound': lower_pct,
                       '% upper bound': upper_pct,
                       'source parameter': ('-rise_mu' if group == 'tic_up' and measure == 'peak' else parameter),
                       'convention': ('session median T_k minus rise_mu, relative to last lick' if group == 'tic_up' and measure == 'peak'
                                      else 'decay_mu, relative to last lick' if group == 'tic_down' and measure == 'peak'
                                      else ''),
                       'comparison': 'reference' if group == 'time_cell' else 'Time cell',
                       'p for median': p_value, 'p for SD': sd_value}
                if measure == 'ramp rate' and group in ['tic_down', 'tic_up']:
                    row.update({'published value': 2.25 if group == 'tic_down' else 1.03,
                                'citation': 'published ramp benchmark'})
                rows.append(row)
        for degree in [1, 4]:
            cells = polynomial[polynomial.group == group]
            rows.append({'group': label, 'measure': f'polynomial degree {degree}', 'n cells': len(cells),
                         'n sessions': cells.session.nunique(), **distribution(cells[f'poly{degree}_trial_mean_ll']),
                         'comparison': 'mean held-out log-likelihood per observed bin, equal trial weights'})
    peak_cells = {}
    for group in GROUPS:
        peak_cells[group] = parameter_values(parameters, group, 'peak', include_bound=True)[0]
    cross_left = peak_cells['tic_down']
    cross_right = peak_cells['tic_up']
    cross_pairs = paired_parameters(cross_left, cross_right, parameter_name('tic_down', 'peak'), parameter_name('tic_up', 'peak'))[2]
    _, cross_p = paired_test(cross_pairs.median_group - cross_pairs.median_time_cell)
    rows.append({'group': 'Ramp-down TIC vs Ramp-up TIC', 'measure': 'peak', 'n cells': len(cross_left) + len(cross_right),
                 'n sessions': len(cross_pairs), 'median': np.nan, 'IQR low': np.nan, 'IQR high': np.nan,
                 'SD': np.nan, 'comparison': 'shared last-lick reference', 'p for median': cross_p,
                 'convention': 'ramp-down decay_mu; ramp-up session median T_k minus rise_mu'})
    for record in cross_pairs.to_dict('records'):
        pair_rows.append(dict(group='Ramp-down TIC vs Ramp-up TIC', measure='peak', bound_subset='included',
                              convention='ramp-down decay_mu; ramp-up session median T_k minus rise_mu', **record))
    table = pd.DataFrame(rows)
    for column in BASE_COLUMNS:
        if column not in table:
            table[column] = np.nan
    table = table[BASE_COLUMNS + [c for c in table if c not in BASE_COLUMNS]]
    table.to_csv(OUTPUT / 'summary_table_wider.csv', index=False)
    pd.DataFrame(pair_rows).to_csv(OUTPUT / 'summary_parameter_session_pairs_wider.csv', index=False)
    figure_table = table[table['bound subset'].isna() | table['bound subset'].eq('included')]
    annotations = make_figure(figure_table, parameters, suffix='_wider', wider=True)
    saved = pd.read_csv(OUTPUT / 'summary_table_wider.csv')
    for annotation in annotations:
        row = saved[(saved.group == annotation['group']) & (saved.measure == annotation['measure'])]
        if annotation['measure'] in ['peak', 'width', 'ramp rate'] and annotation['group'] != 'Ramp-down TIC vs Ramp-up TIC':
            row = row[row['bound subset'] == 'included']
        assert len(row) == 1
        np.testing.assert_allclose(row.iloc[0]['p for median'], annotation['p'], rtol=0, atol=1e-14)
        assert p_label(row.iloc[0]['p for median']) == annotation['text']
    wider_outputs = {'summary_table_wider.csv', 'summary_parameter_session_pairs_wider.csv',
                     'summary_figure_wider_checks.json', 'fig_summary_wider.png', 'fig_summary_wider.pdf',
                     'receptive_field_model_summary.png', 'receptive_field_model_summary.pdf'}
    for path, metadata in protected.items():
        if Path(path).name in wider_outputs:
            continue
        stat = Path(path).stat()
        assert (stat.st_size, stat.st_mtime_ns) == metadata, path
    report = dict(seconds=time.perf_counter() - started, interpreter=sys.executable, old_files_unchanged=True,
                  figure_annotations=annotations, every_figure_p_matches_table=True,
                  width_and_tau='all cells in figure; bound cells marked with crosses; bound-excluded table rows only',
                  peak='all cells shown with bound cells marked; statistics exclude bound cells',
                  session_tests='two-sided paired session medians and SDs; all sessions with finite group medians',
                  polynomials='only the authorised degrees one and four; no other models fitted',
                  literature='published ramp references retained; no new exponent or slope estimated')
    (OUTPUT / 'summary_figure_wider_checks.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wider', action='store_true')
    parser.add_argument('--figure-only', action='store_true',
                        help='redraw saved tables without recomputing statistical summaries')
    args = parser.parse_args()
    if args.figure_only:
        table = pd.read_csv(OUTPUT / ('summary_table_wider.csv' if args.wider else 'summary_table.csv'))
        if args.wider:
            parameters = pd.concat([pd.read_csv(path) for path in sorted(
                (ROOT / 'outputs' / 'undecided_check').glob('*_wider_parameters.csv'))], ignore_index=True)
            table = table[table['bound subset'].isna() | table['bound subset'].eq('included')]
        else:
            parameters = pd.read_csv(FITS / 'fits_all.csv')
        make_figure(table, parameters, wider=args.wider)
    else:
        wider_main() if args.wider else main()
