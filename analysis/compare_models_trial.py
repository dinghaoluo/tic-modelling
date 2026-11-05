'''Created on 5 November 2026
compare saved single-trial predictions with one flat-residual variance per cell
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
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.trial_error_scores import trial_mse
from analysis.compare_models import GROUPS, EXCLUDED, paired_test, markdown

#%% constants
OUTPUT = ROOT / 'outputs' / 'model_comparison'
FIGURES = ROOT / 'figures' / 'model_comparison'
FITS = ROOT / 'outputs' / 'fits_v1_2'
MODELS = ['flat', 'gaussian', 'decay', 'rise', 'poly4']
CONTRASTS = [('gaussian', 'decay'), ('gaussian', 'rise'), ('decay', 'rise')]
PRIMARY = {('tic_down', 'gaussian_minus_decay'), ('tic_up', 'gaussian_minus_rise'),
           ('time_cell', 'gaussian_minus_decay')}

#%% likelihoods
def trial_loglikelihood(mse, variance):
    '''mean Gaussian log-likelihood per bin, with the same variance for every model'''
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError('flat residual variance is not finite and positive')
    return -.5 * (np.log(2 * np.pi * variance) + np.asarray(mse) / variance)


def trial_contrast(mse_a, mse_b, variance):
    '''positive favours model a; the shared Gaussian constant cancels exactly'''
    return (np.asarray(mse_b) - np.asarray(mse_a)) / (2 * variance)


#%% session checks, before any likelihood comparison
def check_session(session):
    started = time.perf_counter()
    cells = pd.read_csv(FITS / 'fits_all.csv')
    cells = cells[cells.session == session].reset_index(drop=True)
    with np.load(ROOT / 'data' / 'binned_v1_1' / f'{session}.npz') as data:
        signal = data['dff']
        trial_rows = data['trial_rows']
    np.testing.assert_array_equal(cells.neu_id, np.arange(1, len(signal) + 1),
                                  err_msg=f'{session}: neuron order does not match the prediction axis')
    folds = np.load(FITS / f'{session}_folds.npz')
    np.testing.assert_array_equal(trial_rows, folds['trial_rows'])
    if len(folds['folds']) != signal.shape[1]:
        raise ValueError(f'{session}: fold count does not match trials')
    bootstrap = folds['bootstrap_trials']
    if bootstrap.shape[1] != signal.shape[1] or np.any(bootstrap < 0) or np.any(bootstrap >= signal.shape[1]):
        raise ValueError(f'{session}: bootstrap indices do not match trials')
    errors = np.empty((len(signal), len(MODELS), signal.shape[1]))
    variances = np.empty(len(signal))
    maximum_difference = 0.
    with h5py.File(FITS / f'{session}_oof.h5', 'r') as saved:
        np.testing.assert_array_equal(saved['trial_rows'][:], trial_rows + 1)
        for model in MODELS:
            if saved[model].shape != signal.shape:
                raise ValueError(f'{session}: prediction dimensions do not match observations')
        for neuron, observed in enumerate(signal):
            for j, model in enumerate(MODELS):
                prediction = saved[model][neuron]
                np.testing.assert_array_equal(np.isfinite(prediction), np.isfinite(observed))
                errors[neuron, j] = trial_mse(observed, prediction)
                difference = abs(errors[neuron, j].mean() - cells.iloc[neuron][f'{model}_held_mse'])
                if not np.isfinite(difference) or difference > 1e-8:
                    raise ValueError(f'{session}, neuron {neuron + 1}, {model}: saved MSE mismatch {difference}')
                maximum_difference = max(maximum_difference, difference)
                if model == 'flat':
                    residual = (observed - prediction)[np.isfinite(observed)]
                    variances[neuron] = np.var(residual, ddof=0)
    if not np.isfinite(variances).all() or np.any(variances <= 0):
        raise ValueError(f'{session}: invalid flat residual variance')
    print(session, 'checks passed', flush=True)
    return dict(session=session, cells=cells[['session', 'neu_id', 'group']].to_dict('records'),
                errors=errors, variances=variances, bootstrap=bootstrap,
                maximum_mse_difference=maximum_difference, check_seconds=time.perf_counter() - started)


def compare_session(checked):
    rows, likelihood_rows = [], []
    resamples = checked['bootstrap']
    for neuron, cell in enumerate(checked['cells']):
        variance = checked['variances'][neuron]
        errors = dict(zip(MODELS, checked['errors'][neuron]))
        likelihood_rows.append(dict(**cell, flat_residual_variance=variance,
            **{f'{model}_trial_mean_ll': trial_loglikelihood(errors[model], variance).mean() for model in MODELS}))
        for a, b in CONTRASTS:
            difference = trial_contrast(errors[a], errors[b], variance)
            lower, upper = np.percentile(difference[resamples].mean(axis=1), [2.5, 97.5])
            rows.append(dict(**cell, contrast=f'{a}_minus_{b}', model_a=a, model_b=b,
                             mean_contrast=difference.mean(), lower=lower, upper=upper,
                             decision=a if lower > 0 else b if upper < 0 else 'undecided',
                             primary=(cell['group'], f'{a}_minus_{b}') in PRIMARY))
    return rows, likelihood_rows


#%% population and session reporting
def summary_row(cells):
    medians = cells.groupby('session').mean_contrast.median()
    statistic, p = paired_test(medians)
    a, b = cells.iloc[0][['model_a', 'model_b']]
    return dict(n_cells=len(cells), median_contrast=cells.mean_contrast.median(),
                pct_above_zero=100 * (cells.mean_contrast > 0).mean(),
                pct_favour_a=100 * (cells.decision == a).mean(),
                pct_favour_b=100 * (cells.decision == b).mean(),
                pct_undecided=100 * (cells.decision == 'undecided').mean(),
                n_sessions=len(medians), n_sessions_positive_median=int((medians > 0).sum()),
                wilcoxon_statistic=statistic, wilcoxon_p=p)


def population_table(frame):
    rows = []
    for subset, selected in [('all_sessions', frame),
                             ('excluded_sessions', frame[~frame.session.isin(EXCLUDED)])]:
        for group in GROUPS:
            for a, b in CONTRASTS:
                contrast = f'{a}_minus_{b}'
                cells = selected[(selected.group == group) & (selected.contrast == contrast)]
                common = dict(subset=subset, group=group, contrast=contrast, model_a=a, model_b=b,
                              primary=(group, contrast) in PRIMARY)
                rows.append(dict(**common, level='group', session='', **summary_row(cells)))
                for session, part in cells.groupby('session'):
                    values = summary_row(part)
                    values['wilcoxon_statistic'] = np.nan
                    values['wilcoxon_p'] = np.nan
                    rows.append(dict(**common, level='session', session=session, **values))
    return pd.DataFrame(rows)


#%% width slopes and figure
def width_slopes(frame):
    cells = frame[frame.group == 'time_cell']
    rows = []
    for subset, selected in [('all_cells', cells), ('positive_mu', cells[cells.gaussian_mu > 0]),
                             ('field_only', cells[cells.gaussian_class == 'field'])]:
        if len(selected) < 3 or selected.gaussian_mu.nunique() < 2:
            raise ValueError(f'{subset}: width slope is not estimable')
        regression = linregress(selected.gaussian_mu, selected.gaussian_sigma)
        rows.append(dict(metric='width_on_peak_slope_trial_check', group='time_cell',
                         parameter='gaussian_sigma_on_mu', subset=subset, n_cells=len(selected),
                         our_value=regression.slope, our_standard_error=regression.stderr,
                         published_value=.14, published_error=.05,
                         citation='Cao et al. (2022), eLife, 11, e75353; https://doi.org/10.7554/eLife.75353'))
    return pd.DataFrame(rows)


def plot_comparison(frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13., 4.))
    colours = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#999999']
    for ax, (a, b) in zip(axes, CONTRASTS):
        for i, (group, colour) in enumerate(zip(GROUPS, colours)):
            cells = frame[(frame.group == group) & (frame.contrast == f'{a}_minus_{b}')]
            medians = cells.groupby('session').mean_contrast.median().to_numpy()
            ax.scatter(i + np.linspace(-.15, .15, len(medians)), medians, s=18, color=colour)
        ax.axhline(0, color='black', linewidth=.7)
        ax.set(xticks=range(len(GROUPS)), xticklabels=GROUPS,
               ylabel=f'{a} minus {b}\nmean held-out log-likelihood per bin')
        ax.tick_params(axis='x', rotation=40)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_model_comparison_trial.png', dpi=180)
    plt.close(fig)


#%% main
def main():
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fits = pd.read_csv(FITS / 'fits_all.csv')
    if fits.duplicated(['session', 'neu_id']).any():
        raise ValueError('duplicate neuron identifiers in saved fits')
    sessions = fits.session.drop_duplicates().tolist()
    workers = min(12, os.cpu_count() or 1, len(sessions))
    print('workers:', workers, flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        checks = list(pool.map(check_session, sessions))
        print('all session MSE and neuron-order checks passed before comparison', flush=True)
        results = list(pool.map(compare_session, checks))
    cells = pd.DataFrame([row for result, _ in results for row in result])
    scores = pd.DataFrame([row for _, result in results for row in result])
    population = population_table(cells)
    slopes = width_slopes(fits)
    cells.to_csv(OUTPUT / 'cell_ll_trial.csv', index=False)
    scores.to_csv(OUTPUT / 'cell_model_ll_trial.csv', index=False)
    population.to_csv(OUTPUT / 'population_ll_trial.csv', index=False)
    literature = pd.read_csv(OUTPUT / 'literature.csv')
    literature = literature[literature.metric != 'width_on_peak_slope_trial_check']
    pd.concat([literature, slopes], ignore_index=True).to_csv(OUTPUT / 'literature.csv', index=False)
    plot_comparison(cells)
    issues = [f'width slope {row.subset} is negative: {row.our_value:.6g}'
              for row in slopes.itertuples() if row.our_value < 0]
    previous = (OUTPUT / 'summary.md').read_text()
    marker = '# Saved-model comparison'
    if marker not in previous:
        raise ValueError('existing summary section not found')
    previous = previous[previous.index(marker):]
    prefix = '# Single-trial held-out likelihood comparison\n\n'
    prefix += ('This table supersedes the mean-trace likelihood table below. All cells are retained, including unsuccessful fit flags. '
               'One variance per cell is the population variance of its held-out flat residuals over observed bins; '
               'that same variance is used for every model. Each trial contributes its mean Gaussian log-likelihood per observed bin, '
               'and each cell contrast is the mean over trials. Percentile intervals resample trials using the saved bootstrap indices. '
               'Positive intervals favour model a, negative intervals model b, and intervals including zero are undecided. '
               'No likelihood-ratio tests are used. Group Wilcoxon tests use session medians, not bins or cells. '
               'Primary contrasts are marked explicitly. Single-session rows have no Wilcoxon test.\n\n')
    prefix += '## Group results\n\n' + markdown(population[population.level == 'group']) + '\n\n'
    prefix += '## Session results\n\n' + markdown(population[population.level == 'session']) + '\n\n'
    prefix += '## Additional time-cell width slopes\n\n' + markdown(slopes) + '\n\n'
    prefix += '## Rows flagged for review\n\n' + ('\n'.join(issues) if issues else 'None.') + '\n\n'
    (OUTPUT / 'summary.md').write_text(prefix + previous)
    report = dict(seconds=time.perf_counter() - started, workers=workers, cells=len(fits),
                  sessions=len(sessions), maximum_mse_difference=max(item['maximum_mse_difference'] for item in checks),
                  all_checks_passed=True, issues=issues, interpreter=sys.executable,
                  variance='np.var of flat residuals over all observed bins, ddof=0',
                  session_checks=[{key: value for key, value in item.items() if key in
                                   ['session', 'maximum_mse_difference', 'check_seconds']} for item in checks])
    (OUTPUT / 'run_report_trial.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
