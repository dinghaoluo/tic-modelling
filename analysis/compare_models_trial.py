# -*- coding: utf-8 -*-
'''
Created on Thu Nov  5 17:47:53 2026

compare saved single-trial predictions with one flat-residual variance per cell

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
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.trial_error_scores import trial_mse
from analysis.compare_models import GROUPS, EXCLUDED, paired_test

#%% constants
OUTPUT = ROOT / 'outputs' / 'model_comparison'
FIGURES = ROOT / 'figures' / 'model_comparison'
FITS = ROOT / 'outputs' / 'fits'
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
    cells = pd.read_csv(FITS / 'fits_all.csv')
    cells = cells[cells.session == session].reset_index(drop=True)
    with np.load(ROOT / 'data' / 'binned' / f'{session}.npz') as data:
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
                maximum_mse_difference=maximum_difference)


def compare_session(checked):
    rows, likelihood_rows = [], []
    resamples = checked['bootstrap']
    for neuron, cell in enumerate(checked['cells']):
        variance = checked['variances'][neuron]
        errors = dict(zip(MODELS, checked['errors'][neuron]))
        likelihood_rows.append(dict(**cell, flat_residual_variance=variance,
            **{f'{model}_trial_mean_ll': trial_loglikelihood(errors[model], variance).mean() for model in MODELS}))
        for model_a, model_b in CONTRASTS:
            difference = trial_contrast(errors[model_a], errors[model_b], variance)
            lower, upper = np.percentile(difference[resamples].mean(axis=1), [2.5, 97.5])
            rows.append(dict(**cell, contrast=f'{model_a}_minus_{model_b}', model_a=model_a, model_b=model_b,
                             mean_contrast=difference.mean(), lower=lower, upper=upper,
                             decision=model_a if lower > 0 else model_b if upper < 0 else 'undecided',
                             primary=(cell['group'], f'{model_a}_minus_{model_b}') in PRIMARY))
    return rows, likelihood_rows


#%% population and session tables
def summary_row(cells):
    medians = cells.groupby('session').mean_contrast.median()
    statistic, p_value = paired_test(medians)
    model_a, model_b = cells.iloc[0][['model_a', 'model_b']]
    return dict(n_cells=len(cells), median_contrast=cells.mean_contrast.median(),
                pct_above_zero=100 * (cells.mean_contrast > 0).mean(),
                pct_favour_a=100 * (cells.decision == model_a).mean(),
                pct_favour_b=100 * (cells.decision == model_b).mean(),
                pct_undecided=100 * (cells.decision == 'undecided').mean(),
                n_sessions=len(medians), n_sessions_positive_median=int((medians > 0).sum()),
                wilcoxon_statistic=statistic, wilcoxon_p=p_value)


def population_table(frame):
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
    for ax, (model_a, model_b) in zip(axes, CONTRASTS):
        for i, (group, colour) in enumerate(zip(GROUPS, colours)):
            cells = frame[(frame.group == group) & (frame.contrast == f'{model_a}_minus_{model_b}')]
            medians = cells.groupby('session').mean_contrast.median().to_numpy()
            ax.scatter(i + np.linspace(-.15, .15, len(medians)), medians, s=18, color=colour)
        ax.axhline(0, color='black', linewidth=.7)
        ax.set(xticks=range(len(GROUPS)), xticklabels=GROUPS,
               ylabel=f'{model_a} minus {model_b}\nmean held-out log-likelihood per bin')
        ax.tick_params(axis='x', rotation=40)
    fig.tight_layout()
    fig.savefig(FIGURES / 'fig_model_comparison_trial.png', dpi=180)
    plt.close(fig)


#%% main
def main():
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


if __name__ == '__main__':
    main()
