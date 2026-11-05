'''Created on 5 November 2026
assemble the requested held-out comparisons and parameter summary figure
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
from utils.trial_error_scores import trial_mse
from analysis.compare_models_trial import trial_loglikelihood, trial_contrast
from analysis.compare_models import paired_test

#%% constants
OUTPUT = ROOT / 'outputs' / 'model_comparison'
FIGURES = ROOT / 'figures' / 'model_comparison'
FITS = ROOT / 'outputs' / 'fits_v1_2'
GROUPS = ['tic_down', 'tic_up', 'time_cell']
LABELS = ['Ramp-down TIC', 'Ramp-up TIC', 'Time cell']
COLOURS = ['#4477AA', '#EE6677', '#228833']
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


def parameter_values(fits, group, measure):
    parameter = {'peak': 'gaussian_mu', 'width': 'gaussian_sigma',
                 'ramp rate': 'rise_tau' if group == 'tic_up' else 'decay_tau'}[measure]
    cells = fits[fits.group == group]
    return cells[~cells[f'{parameter}_at_bound']], parameter, 100 * cells[f'{parameter}_at_bound'].mean()


def paired_parameters(left, right, left_parameter, right_parameter):
    pairs = []
    for session in left.session.drop_duplicates():
        a = left.loc[left.session == session, left_parameter]
        b = right.loc[right.session == session, right_parameter]
        if len(a) < 5 or len(b) < 5:
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
                   'source parameter': parameter}
            if group != 'time_cell':
                reference_cells, reference_parameter, _ = parameter_values(fits, 'time_cell', measure)
                p, sd_p, pairs = paired_parameters(retained, reference_cells, parameter, reference_parameter)
                row.update({'n sessions': len(pairs), 'p for median': p, 'p for SD': sd_p})
                pair_rows.extend([dict(group=label, measure=measure, **record) for record in pairs.to_dict('records')])
            if measure == 'width' and group == 'time_cell':
                row.update({'published value': .69, 'citation': 'Cao et al. 2022, eLife; https://doi.org/10.7554/eLife.75353'})
            if measure == 'ramp rate' and group in ['tic_down', 'tic_up']:
                row.update({'published value': 2.25 if group == 'tic_down' else 1.03,
                            'citation': 'Cao et al. 2024, PNAS; https://sites.bu.edu/tcn/files/2024/09/Cao_et_al_2024.pdf'})
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
        row = {'group': 'Time cell', 'measure': measure, 'n cells': value.n_cells,
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
    table = pd.DataFrame(rows)
    for column in BASE_COLUMNS:
        if column not in table:
            table[column] = np.nan
    return table[BASE_COLUMNS + [column for column in table if column not in BASE_COLUMNS]], pd.DataFrame(pair_rows)


#%% figure
def make_figure(table, fits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12,
                         'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
                         'pdf.fonttype': 42})
    fig = plt.figure(figsize=(16., 11.))
    grid = fig.add_gridspec(2, 6, height_ratios=[1., 1.15], hspace=.55, wspace=1.)
    axes = [fig.add_subplot(grid[0, :3]), fig.add_subplot(grid[0, 3:]),
            fig.add_subplot(grid[1, :2]), fig.add_subplot(grid[1, 2:4]), fig.add_subplot(grid[1, 4:])]
    annotations = []
    for ax, panel, title, categories in [
            (axes[0], 'A', 'Gaussian vs exponential', ['Gaussian better', 'Exponential better', 'No clear difference']),
            (axes[1], 'B', 'Linear vs curved', ['Curved better', 'Linear better', 'No clear difference'])]:
        selected = table[table.measure == title].set_index('group').loc[LABELS]
        bottom = np.zeros(len(GROUPS))
        for category, colour in zip(categories, ['#4477AA', '#EE6677', '#BBBBBB']):
            values = selected['% ' + category].to_numpy(float)
            ax.bar(np.arange(len(GROUPS)), values, bottom=bottom, color=colour, width=.65, label=category)
            bottom += values
        np.testing.assert_allclose(bottom, 100., atol=1e-10)
        for i, (label, row) in enumerate(selected.iterrows()):
            text = p_label(row['p for median'])
            ax.text(i, 103, text, ha='center', va='bottom', fontsize=11)
            annotations.append(dict(panel=panel, group=label, measure=title, p=float(row['p for median']), text=text))
        ax.set(title=title, ylabel='Cells (%)', ylim=(0, 114), yticks=np.arange(0, 101, 20),
               xticks=range(len(LABELS)), xticklabels=[f'{label}\n(n = {int(selected.loc[label, "n cells"])})' for label in LABELS])
        ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.28), ncol=3)
    for ax, panel, measure, title, ylabel in zip(axes[2:], ['C', 'D', 'E'],
            ['peak', 'width', 'ramp rate'], ['Peak time', 'Width', 'Ramp rate'],
            ['Peak time (s)', 'Width (s)', 'Time constant (s)']):
        values = [parameter_values(fits, group, measure)[0][parameter_values(fits, group, measure)[1]].to_numpy()
                  for group in GROUPS]
        boxes = ax.boxplot(values, tick_labels=LABELS, showfliers=False, patch_artist=True,
                           medianprops={'color': 'black', 'linewidth': 1.5})
        for box, colour in zip(boxes['boxes'], COLOURS):
            box.set(facecolor=colour, alpha=.55)
        ax.set(title=title, ylabel=ylabel)
        ax.tick_params(axis='x', rotation=20)
        low, high = ax.get_ylim()
        span = high - low
        for i, label in enumerate(LABELS[:2]):
            row = table[(table.group == label) & (table.measure == measure)].iloc[0]
            height = high + (.08 + .19 * i) * span
            ax.plot([i + 1, i + 1, 3, 3], [height, height + .035 * span, height + .035 * span, height],
                    color='black', linewidth=.8)
            text = p_label(row['p for median'])
            ax.text((i + 1 + 3) / 2, height + .05 * span, text, ha='center', va='bottom', fontsize=10)
            annotations.append(dict(panel=panel, group=label, measure=measure, p=float(row['p for median']), text=text))
        ax.set_ylim(low, high + .47 * span)
        if measure == 'width':
            ax.hlines(.69, 2.78, 3.22, colors='#AA4499', linewidth=2.2)
            ax.legend(handles=[Line2D([0], [0], color='#AA4499', linewidth=2.2,
                       label='Cao et al. 2022 time cells')], frameon=False, loc='upper left', fontsize=9)
        if measure == 'ramp rate':
            ax.hlines(2.25, .78, 1.22, colors='#AA4499', linewidth=2.2)
            ax.hlines(1.03, 1.78, 2.22, colors='#CCBB44', linewidth=2.2)
            ax.legend(handles=[Line2D([0], [0], color='#AA4499', linewidth=2.2, label='Cao et al. 2024 past'),
                               Line2D([0], [0], color='#CCBB44', linewidth=2.2, label='Cao et al. 2024 future')],
                      frameon=False, loc='upper left', fontsize=9)
    for ax, letter in zip(axes, 'ABCDE'):
        ax.text(-.12, 1.05, letter, transform=ax.transAxes, fontsize=16, fontweight='bold', ha='right')
    fig.subplots_adjust(left=.065, right=.98, bottom=.09, top=.88)
    fig.savefig(FIGURES / 'fig_summary.png', dpi=200)
    fig.savefig(FIGURES / 'fig_summary.pdf')
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


if __name__ == '__main__':
    main()
