'''Created on 5 November 2026
compare saved likelihoods and fitted parameter distributions without refitting
@author: Dinghao Luo
'''

#%% imports
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.summarise_fits_v1_2 import earlier

#%% constants
INPUT   = ROOT / 'outputs' / 'fits_v1_2' / 'fits_all.csv'
OUTPUT  = ROOT / 'outputs' / 'model_comparison'
FIGURES = ROOT / 'figures' / 'model_comparison'
GROUPS  = ['tic_down', 'tic_up', 'time_cell', 'tic_and_time_cell', 'other']
PARAMS  = ['gaussian_mu', 'gaussian_sigma', 'decay_mu', 'decay_tau', 'rise_mu', 'rise_tau']
MODELS  = ['flat', 'gaussian', 'decay', 'rise', 'poly4']
EXCLUDED = ['A132-20221124-02', 'A145-20230313-02']
MATCHED = [('tic_down', 'decay_tau', 'past'), ('tic_up', 'rise_tau', 'future'),
           ('time_cell', 'decay_tau', 'past')]
CAO_2024 = 'Cao, Bright & Howard (2024), PNAS; https://sites.bu.edu/tcn/files/2024/09/Cao_et_al_2024.pdf'
CAO_2022 = 'Cao et al. (2022), eLife, 11, e75353; https://doi.org/10.7554/eLife.75353'

#%% contrasts and bounds
def likelihood_contrasts(frame):
    '''positive differences favour the Gaussian; choose the exponential separately per cell'''
    gaussian = frame.gaussian_held_ll.to_numpy()
    decay = frame.decay_held_ll.to_numpy()
    rise = frame.rise_held_ll.to_numpy()
    return pd.DataFrame({'gaussian_minus_decay': gaussian - decay,
                         'gaussian_minus_rise': gaussian - rise,
                         'gaussian_minus_better_exponential': gaussian - np.maximum(decay, rise)},
                        index=frame.index)


def shape_bound(frame, shape):
    width = 'sigma' if shape == 'gaussian' else 'tau'
    return (frame[f'{shape}_mu_at_bound'] | frame[f'{shape}_{width}_at_bound'] |
            frame[f'{shape}_a_at_bound'])


def paired_test(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.nan
    if np.all(values == 0):
        return 0., 1.
    result = wilcoxon(values, alternative='two-sided')
    return float(result.statistic), float(result.pvalue)


def stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(median=np.nan, q25=np.nan, q75=np.nan, iqr=np.nan, variance=np.nan, sd=np.nan)
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    variance = np.var(values, ddof=1) if len(values) > 1 else np.nan
    return dict(median=median, q25=q25, q75=q75, iqr=q75 - q25,
                variance=variance, sd=np.sqrt(variance))


#%% population likelihoods
def population_table(frame):
    rows, session_rows = [], []
    for subset, selected in [('all_sessions', frame),
                             ('excluded_sessions', frame[~frame.session.isin(EXCLUDED)])]:
        for group in GROUPS:
            cells = selected[selected.group == group]
            contrasts = likelihood_contrasts(cells)
            benchmarks = {f'{model}_median_held_ll': cells[f'{model}_held_ll'].median() for model in MODELS}
            for contrast in contrasts:
                values = contrasts[contrast]
                medians = values.groupby(cells.session).median()
                statistic, p = paired_test(medians)
                rows.append(dict(subset=subset, group=group, contrast=contrast, n_cells=len(cells),
                                 median_difference=values.median(), pct_above_zero=100 * (values > 0).mean(),
                                 summed_difference=values.sum(), n_sessions=len(medians),
                                 n_sessions_positive_median=int((medians > 0).sum()),
                                 wilcoxon_statistic=statistic, wilcoxon_p=p, **benchmarks))
                session_rows.extend([dict(subset=subset, group=group, contrast=contrast,
                                          session=session, median_difference=value,
                                          n_cells=int((cells.session == session).sum()))
                                     for session, value in medians.items()])
    return pd.DataFrame(rows), pd.DataFrame(session_rows)


#%% parameter distributions and paired sessions
def parameter_table(frame):
    rows, pairs = [], []
    for group in GROUPS:
        cells = frame[frame.group == group]
        for parameter in PARAMS:
            shape = parameter.split('_')[0]
            any_bound = shape_bound(cells, shape)
            modes = ['included', 'excluded'] if parameter.endswith('_tau') else ['included']
            for mode in modes:
                kept = cells if mode == 'included' else cells[~any_bound]
                rows.append(dict(analysis='distribution', group=group, parameter=parameter,
                                 bounds=mode, n_cells=len(kept), n_before=len(cells),
                                 pct_at_bound=100 * kept[f'{parameter}_at_bound'].mean() if len(kept) else np.nan,
                                 pct_any_shape_bound=100 * any_bound.mean(), **stats(kept[parameter])))
        for kind in ['field', 'monotonic']:
            rows.append(dict(analysis='gaussian_class', group=group, parameter=kind, n_cells=len(cells),
                             count=int((cells.gaussian_class == kind).sum()),
                             percentage=100 * (cells.gaussian_class == kind).mean()))
        for shape, width in [('gaussian', 'sigma'), ('decay', 'tau')]:
            kept = cells[~shape_bound(cells, shape)]
            result = spearmanr(kept[f'{shape}_mu'], kept[f'{shape}_{width}']) if len(kept) >= 2 else None
            rows.append(dict(analysis='spearman', group=group, parameter=f'{shape}_mu_vs_{width}',
                             bounds='excluded', n_cells=len(kept),
                             correlation=result.statistic if result is not None else np.nan,
                             spearman_p=result.pvalue if result is not None else np.nan))
    for group in ['tic_down', 'tic_up']:
        for parameter in ['gaussian_mu', 'gaussian_sigma', 'matched_tau']:
            a_column = ('decay_tau' if group == 'tic_down' else 'rise_tau') if parameter == 'matched_tau' else parameter
            b_column = 'decay_tau' if parameter == 'matched_tau' else parameter
            modes = ['included', 'excluded'] if parameter == 'matched_tau' else ['included']
            for mode in modes:
                left = frame[frame.group == group]
                right = frame[frame.group == 'time_cell']
                if mode == 'excluded':
                    left = left[~shape_bound(left, a_column.split('_')[0])]
                    right = right[~shape_bound(right, b_column.split('_')[0])]
                for metric in ['median', 'sd']:
                    differences, a_values, b_values, sessions = [], [], [], []
                    for session in frame.session.drop_duplicates():
                        a = left[left.session == session][a_column]
                        b = right[right.session == session][b_column]
                        if len(a) < 5 or len(b) < 5:
                            continue
                        a_value, b_value = stats(a)[metric], stats(b)[metric]
                        differences.append(a_value - b_value)
                        a_values.append(a_value)
                        b_values.append(b_value)
                        sessions.append(session)
                        pairs.append(dict(group=group, comparison_group='time_cell', parameter=parameter,
                                          group_parameter=a_column, comparison_parameter=b_column, bounds=mode,
                                          metric=metric, session=session, n_group=len(a), n_time_cell=len(b),
                                          group_value=a_value, time_cell_value=b_value, difference=a_value - b_value))
                    statistic, p = paired_test(differences)
                    rows.append(dict(analysis='paired_session_wilcoxon', group=group, comparison_group='time_cell',
                                     parameter=parameter, group_parameter=a_column, comparison_parameter=b_column,
                                     bounds=mode, metric=metric, n_sessions=len(sessions),
                                     group_session_value_median=np.median(a_values) if a_values else np.nan,
                                     time_cell_session_value_median=np.median(b_values) if b_values else np.nan,
                                     session_difference_median=np.median(differences) if differences else np.nan,
                                     wilcoxon_statistic=statistic, wilcoxon_p=p, sessions=';'.join(sessions)))
    return pd.DataFrame(rows), pd.DataFrame(pairs)


#%% published benchmarks
def literature_table(frame, published):
    cells = frame[frame.group == 'time_cell']
    sessions = frame.session.drop_duplicates().to_numpy()
    session_values = {session: cells.loc[cells.session == session, 'gaussian_mu'].to_numpy() for session in sessions}
    alpha = earlier.power_exponent(cells.gaussian_mu.to_numpy())
    rng = np.random.default_rng(20261104)
    samples = [earlier.power_exponent(np.concatenate([session_values[session] for session in
               rng.choice(sessions, len(sessions), replace=True)])) for _ in range(1000)]
    lower, upper = np.percentile(samples, [2.5, 97.5])
    regression = linregress(cells.gaussian_mu, cells.gaussian_sigma)
    rows = []
    for group, parameter, direction in MATCHED:
        part = frame[frame.group == group]
        for mode in ['included', 'excluded']:
            kept = part if mode == 'included' else part[~shape_bound(part, parameter.split('_')[0])]
            values = stats(kept[parameter])
            rows.append(dict(metric='matched_tau_median', group=group, parameter=parameter, bounds=mode,
                             n_cells=len(kept), our_value=values['median'], our_q25=values['q25'], our_q75=values['q75'],
                             published_value=published[f'{direction}_tau'], citation=CAO_2024))
        shape = parameter.split('_')[0]
        rows.append(dict(metric='mu_median', group=group, parameter=f'{shape}_mu', bounds='included',
                         n_cells=len(part), our_value=part[f'{shape}_mu'].median(),
                         published_value=published[f'{direction}_mu'], citation=CAO_2024))
    for group in [*GROUPS, 'all']:
        part = frame if group == 'all' else frame[frame.group == group]
        better = np.maximum(part.decay_held_ll, part.rise_held_ll) > part.gaussian_held_ll
        rows.append(dict(metric='exponential_better_than_gaussian_pct', group=group, n_cells=len(part),
                         our_count=int(better.sum()), our_value=100 * better.mean(),
                         published_count=published['exponential_count'], published_n=published['exponential_total'],
                         published_value=100 * published['exponential_count'] / published['exponential_total'], citation=CAO_2024))
    rows.extend([
        dict(metric='peak_power_law_alpha', group='time_cell', parameter='gaussian_mu', n_cells=len(cells),
             n_peaks_in_limits=int(cells.gaussian_mu.between(.35, 7.).sum()), our_value=alpha,
             our_lower=lower, our_upper=upper, interval='session-bootstrap percentile 95%',
             published_value=published['alpha'], published_lower=published['alpha_lower'],
             published_upper=published['alpha_upper'], citation=CAO_2022),
        dict(metric='width_on_peak_slope', group='time_cell', parameter='gaussian_sigma_on_mu', n_cells=len(cells),
             our_value=regression.slope, our_standard_error=regression.stderr,
             published_value=published['width_slope'], published_error=published['width_slope_error'], citation=CAO_2022),
        dict(metric='sigma_median', group='time_cell', parameter='gaussian_sigma', n_cells=len(cells),
             our_value=cells.gaussian_sigma.median(), published_value=published['sigma_median'], citation=CAO_2022),
        ])
    return pd.DataFrame(rows)


#%% figures
def make_figures(frame, published):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colours = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#999999']
    differences = likelihood_contrasts(frame).gaussian_minus_better_exponential
    fig, ax = plt.subplots(figsize=(8., 4.))
    values = [differences[frame.group == group].to_numpy() for group in GROUPS]
    ax.boxplot(values, tick_labels=GROUPS, showfliers=False, showmeans=False)
    for i, group in enumerate(GROUPS, 1):
        mask = frame.group == group
        medians = differences[mask].groupby(frame.loc[mask, 'session']).median().to_numpy()
        offsets = np.linspace(-.15, .15, len(medians))
        ax.scatter(i + offsets, medians, s=18, color=colours[i - 1], label='session medians' if i == 1 else None)
    ax.axhline(0, color='black', linewidth=.7)
    ax.set(ylabel='held-out log-likelihood difference\n(Gaussian minus better exponential)')
    ax.tick_params(axis='x', rotation=20)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_model_comparison.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13., 4.))
    for ax, parameter in zip(axes[:2], ['gaussian_mu', 'gaussian_sigma']):
        ax.boxplot([frame.loc[frame.group == group, parameter] for group in GROUPS],
                   tick_labels=GROUPS, showfliers=False, showmeans=False)
        ax.set(ylabel=parameter.replace('_', ' ') + ' (s)')
        ax.tick_params(axis='x', rotation=35)
    values, labels = [], []
    for group, parameter, _ in MATCHED:
        cells = frame[frame.group == group]
        values.append(cells.loc[~shape_bound(cells, parameter.split('_')[0]), parameter])
        labels.append(group)
    axes[2].boxplot(values, tick_labels=labels, showfliers=False, showmeans=False)
    axes[2].set(ylabel='matched tau, not at a shape bound (s)')
    axes[2].tick_params(axis='x', rotation=35)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_parameters.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7., 4.))
    for i, (group, parameter, direction) in enumerate(MATCHED):
        cells = frame[frame.group == group]
        for offset, mode, colour in [(-.12, 'included', '#4477AA'), (.12, 'excluded', '#228833')]:
            kept = cells if mode == 'included' else cells[~shape_bound(cells, parameter.split('_')[0])]
            values = stats(kept[parameter])
            median = values['median']
            ax.errorbar(i + offset, median, yerr=np.array([[median - values['q25']], [values['q75'] - median]]),
                        fmt='o', color=colour, capsize=4, label=f'bounds {mode}' if i == 0 else None)
        ax.scatter(i, published[f'{direction}_tau'], marker='_', s=300, color='#EE6677',
                   label='Cao et al. 2024 median' if i == 0 else None)
    ax.set(xticks=range(len(MATCHED)), xticklabels=[entry[0] for entry in MATCHED], ylabel='matched tau (s)')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_literature.png', dpi=180)
    plt.close(fig)


#%% output checks and report
def markdown(frame):
    def show(value):
        if pd.isna(value):
            return ''
        return f'{value:.6g}' if isinstance(value, (float, np.floating)) else str(value)
    return '\n'.join(['| ' + ' | '.join(frame.columns) + ' |',
                      '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |',
                      *['| ' + ' | '.join(show(value) for value in row) + ' |'
                        for row in frame.itertuples(index=False, name=None)]])


def main():
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT)
    if frame.duplicated(['session', 'neu_id']).any() or set(frame.group) != set(GROUPS):
        raise ValueError('duplicate cells or unexpected groups in source table')
    for column in [*[f'{name}_held_ll' for name in MODELS], *PARAMS]:
        if not np.isfinite(frame[column]).all():
            raise ValueError(f'non-finite required input: {column}')
    if not frame.gaussian_class.isin(['field', 'monotonic']).all():
        raise ValueError('unexpected Gaussian class')
    for shape in ['gaussian', 'decay', 'rise']:
        for column in [name for name in frame if name.startswith(shape + '_') and name.endswith('_at_bound')]:
            if frame[column].isna().any() or not frame[column].isin([True, False]).all():
                raise ValueError(f'invalid bound flag: {column}')
    published = dict(past_mu=.01, past_tau=2.25, future_mu=-.02, future_tau=1.03,
                     exponential_count=647, exponential_total=722, alpha=.92,
                     alpha_lower=.73, alpha_upper=1.11, width_slope=.14,
                     width_slope_error=.05, sigma_median=.69)
    population, medians = population_table(frame)
    parameters, pairs = parameter_table(frame)
    literature = literature_table(frame, published)
    for name, values in [('population_ll', population), ('population_session_medians', medians),
                         ('parameters', parameters), ('parameter_session_pairs', pairs), ('literature', literature)]:
        values.to_csv(OUTPUT / f'{name}.csv', index=False)
    make_figures(frame, published)
    issues = []
    for i, row in parameters.iterrows():
        if row.analysis == 'paired_session_wilcoxon' and row.n_sessions == 0:
            issues.append(f'parameters.csv row {i + 2}: no eligible paired sessions')
        if row.analysis == 'distribution' and row.n_cells == 0:
            issues.append(f'parameters.csv row {i + 2}: no cells after bound exclusion')
        if row.analysis == 'spearman' and not np.isfinite(row.correlation):
            issues.append(f'parameters.csv row {i + 2}: undefined correlation')
    slope = literature[literature.metric == 'width_on_peak_slope'].iloc[0]
    if slope.our_value < 0 < slope.published_value:
        issues.append('literature.csv: width_on_peak_slope is negative while the supplied published slope is positive; retained without filtering or correction')
    summary = '# Saved-model comparison\n\n'
    summary += ('All cells are retained, without filtering on beats_flat, preferred or fit_success. '
                'Likelihoods are the saved mean-trace held-out scores, not trial-wise Gaussian noise likelihoods; '
                'a positive contrast favours the Gaussian. The restricted likelihood table excludes the named sessions only.\n\n'
                'Parameter variance and SD use the sample convention. Individual parameter bound percentages use that parameter flag; '
                'tau exclusion and correlation exclusion use any bound flag of the corresponding shape. '
                'Paired parameter tests use sessions with at least five retained cells in each group, '
                'and matched tau is decay for tic_down and time_cell, rise for tic_up.\n\n'
                'The time-cell peak exponent uses all time_cell rows within the existing peak limits, '
                'with the existing estimator and session-resampling interval. The width slope is ordinary least squares with an intercept '
                'on all time_cell rows. Published benchmarks are descriptive comparisons, not tests of equivalence.\n\n')
    for heading, values in [('Population likelihoods', population), ('Session likelihood medians', medians)]:
        summary += f'## {heading}\n\n' + markdown(values) + '\n\n'
    for analysis, values in parameters.groupby('analysis', sort=False):
        summary += f'## Parameters: {analysis}\n\n' + markdown(values.dropna(axis=1, how='all')) + '\n\n'
    summary += '## Paired parameter session values\n\n' + markdown(pairs) + '\n\n'
    summary += '## Published benchmarks\n\n' + markdown(literature) + '\n\n'
    summary += '## Output checks\n\n' + ('\n'.join(issues) if issues else 'No undefined required output rows were found.') + '\n'
    (OUTPUT / 'summary.md').write_text(summary)
    report = dict(seconds=time.perf_counter() - started, interpreter=sys.executable, input=str(INPUT),
                  cells=len(frame), sessions=int(frame.session.nunique()), processes=1, issues=issues,
                  unfiltered_failed_fit_cells=int((~frame.fit_success).sum()))
    (OUTPUT / 'run_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
