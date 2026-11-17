'''Created on 17 November 2026
compare revised bumps and plain exponentials in the saved wider-window folds
@author: Dinghao Luo
'''

#%% imports
import os
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.two_exponential import contrast_row, signed_test, markdown, predict_bump
from utils.rf_models_v1_2 import linear_amplitude
from utils.trial_error_scores import trial_mse

DATA = ROOT / 'data' / 'binned_v1_3'
WIDER = ROOT / 'outputs' / 'undecided_check'
FIRST = ROOT / 'outputs' / 'two_exponential'
OUTPUT = ROOT / 'outputs' / 'two_exponential_v2'
FIGURES = ROOT / 'figures'
os.environ['MPLCONFIGDIR'] = str(OUTPUT / 'matplotlib')
GROUPS = ['tic_down', 'tic_up', 'time_cell']
PRIMARY = {'tic_down': 'last', 'tic_up': 'next', 'time_cell': 'last'}
MODELS = ['revised_last', 'revised_next', 'plain']
KEYS = {'revised_last': ['t0', 'u', 'tau1', 'tau2'],
        'revised_next': ['t0', 'u', 'tau1', 'tau2'], 'plain': ['mu', 'tau']}
LOWER = {'revised_last': np.array([-1., 0., .1, .1]),
         'revised_next': np.array([-1., 0., .1, .1]), 'plain': np.array([-1., .1])}
UPPER = {'revised_last': np.array([3., 7., 10., 10.]),
         'revised_next': np.array([3., 7., 10., 10.]), 'plain': np.array([3., 10.])}
BUMP_GRID = np.array(list(product([-1., 0., 1., 2., 3.], [0., .5, 1., 2., 3., 4., 5.5, 7.],
                                  np.geomspace(.1, 10., 4), np.geomspace(.1, 10., 4))))
PLAIN_GRID = np.array(list(product(np.linspace(-1., 3., 10), np.geomspace(.1, 10., 12))))

#%% shapes and profiled fits
def peak_and_onset(interval, params, anchoring):
    t0, u = params[:2]
    peak = np.minimum(u, interval) if anchoring == 'last' else np.maximum(interval - u, t0)
    return peak, np.minimum(t0, peak)


def revised_shape(t, interval, params, anchoring, derivatives=False):
    t, interval = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(interval, dtype=float))
    t0, u, tau1, tau2 = params
    peak, onset = peak_and_onset(interval, params, anchoring)
    dp = np.zeros((t.size, 2))
    ds = np.zeros_like(dp)
    if anchoring == 'last':
        dp[:, 1] = (u < interval).ravel()
        ds[:, 0] = (t0 < peak).ravel()
        ds[:, 1] = (t0 >= peak).ravel() * dp[:, 1]
    else:
        dp[:, 0] = (t0 >= interval - u).ravel()
        dp[:, 1] = -(t0 < interval - u).ravel().astype(float)
        ds[:, 0] = 1.
    shape = np.zeros(t.shape)
    jac = np.zeros((t.size, 4))
    for rising, tau in [(True, tau1), (False, tau2)]:
        keep = (t > onset) & (t < interval) & ((t <= peak) if rising else (t > peak))
        x = (t - onset)[keep] if rising else (interval - t)[keep]
        length = (peak - onset)[keep] if rising else (interval - peak)[keep]
        numerator = -np.expm1(-x / tau)
        denominator = -np.expm1(-length / tau)
        shape[keep] = numerator / denominator
        if derivatives:
            dlength = -numerator * np.exp(-length / tau) / (tau * denominator ** 2)
            dtau = (-x * np.exp(-x / tau) * denominator +
                    numerator * length * np.exp(-length / tau)) / (tau ** 2 * denominator ** 2)
            k = keep.ravel()
            if rising:
                dx = np.exp(-x / tau) / (tau * denominator)
                jac[k, :2] = -dx[:, None] * ds[k] + dlength[:, None] * (dp[k] - ds[k])
            else:
                jac[k, :2] = -dlength[:, None] * dp[k]
            jac[k, 2 if rising else 3] = dtau
    return (shape, jac) if derivatives else shape


def plain_shape(t, interval, params, mirrored=False, derivatives=False):
    t, interval = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(interval, dtype=float))
    mu, tau = params
    x = interval - t if mirrored else t
    delay = np.maximum(x - mu, 0.)
    shape = np.exp(-delay / tau)
    jac = np.column_stack([(shape * (x > mu) / tau).ravel(), (shape * delay / tau ** 2).ravel()])
    return (shape, jac) if derivatives else shape


def model_shape(model, t, interval, params, mirrored=False, derivatives=False):
    if model == 'plain':
        return plain_shape(t, interval, params, mirrored, derivatives)
    return revised_shape(t, interval, params, model.removeprefix('revised_'), derivatives)


def predict_model(model, t, interval, params, mirrored=False):
    return params[0] + params[1] * model_shape(model, t, interval, params[2:], mirrored)


def coarse_grid(model, t, interval, weights, mirrored=False):
    nonlinear = PLAIN_GRID if model == 'plain' else BUMP_GRID
    shapes = np.array([model_shape(model, t, interval, p, mirrored) for p in nonlinear])
    centred = shapes - np.average(shapes, axis=1, weights=weights)[:, None]
    return centred, (centred ** 2) @ weights


def fit_model(model, t, interval, y, weights, mirrored=False, grid=None):
    if grid is None:
        grid = coarse_grid(model, t, interval, weights, mirrored)
    centred, denominator = grid
    y_mean = np.average(y, weights=weights)
    covariance = centred @ (weights * (y - y_mean))
    scores = np.divide(np.maximum(covariance, 0.) ** 2, denominator,
                       out=np.zeros_like(covariance), where=denominator > 0)
    nonlinear = PLAIN_GRID if model == 'plain' else BUMP_GRID
    initial = nonlinear[np.argmax(scores)]
    square_root = np.sqrt(weights)
    previous, cached = None, None

    def evaluate(params):
        nonlocal previous, cached
        if previous is not None and np.array_equal(params, previous):
            return cached
        shape, derivative = model_shape(model, t, interval, params, mirrored, True)
        shape_mean = np.average(shape, weights=weights)
        centred_shape = shape - shape_mean
        denom = np.dot(weights, centred_shape ** 2)
        cov = np.dot(weights * centred_shape, y - y_mean)
        amplitude = max(cov / denom, 0.) if denom > 0 else 0.
        derivative_mean = np.average(derivative, axis=0, weights=weights)
        amplitude_derivative = np.zeros(len(initial))
        if amplitude > 0:
            dcov = derivative.T @ (weights * (y - y_mean))
            ddenom = 2. * derivative.T @ (weights * centred_shape)
            amplitude_derivative = (dcov * denom - cov * ddenom) / denom ** 2
        residual = square_root * (y_mean + amplitude * centred_shape - y)
        jacobian = square_root[:, None] * (amplitude * (derivative - derivative_mean) +
                                           centred_shape[:, None] * amplitude_derivative)
        previous = params.copy()
        cached = residual, jacobian
        return cached

    result = least_squares(lambda p: evaluate(p)[0], initial,
                           jac=lambda p: evaluate(p)[1], bounds=(LOWER[model], UPPER[model]), x_scale='jac')
    shape = model_shape(model, t, interval, result.x, mirrored)
    baseline, amplitude = linear_amplitude(y, shape, weights)
    return np.r_[baseline, amplitude, result.x], result.success, result.nfev


#%% held-out comparisons
def session_job(session):
    started = time.perf_counter()
    data = np.load(DATA / f'{session}.npz')
    signals, times, intervals, ids = [data[k] for k in ['dff', 't', 'T_k', 'neu_id']]
    fold_data = np.load(ROOT / 'outputs' / 'fits_v1_2' / f'{session}_folds.npz')
    np.testing.assert_array_equal(data['trial_rows'], fold_data['trial_rows'])
    folds, bootstrap = fold_data['folds'], fold_data['bootstrap_trials']
    cells = pd.read_csv(WIDER / f'{session}_wider_parameters.csv').set_index('neu_id').loc[ids].reset_index()
    prior = pd.read_csv(WIDER / f'{session}_wider_contrasts.csv')
    assert prior.groupby('neu_id').flat_residual_variance.nunique().eq(1).all()
    variances = prior.groupby('neu_id').flat_residual_variance.first()
    failures, parameters, rows, clipping = [], [], [], []
    observed = np.isfinite(signals[0])
    if not np.array_equal(np.isfinite(signals), np.broadcast_to(observed, signals.shape)):
        raise ValueError(f'{session}: cell-specific masks need separate compression')
    with h5py.File(OUTPUT / f'{session}_oof.h5', 'w') as saved:
        saved.create_dataset('neu_id', data=ids)
        saved.create_dataset('trial_rows', data=data['trial_rows'])
        saved.create_dataset('t', data=times)
        for model in MODELS:
            held = np.full(signals.shape, np.nan)
            for stage in [0, 1, 2, 3, 4, 'full']:
                train = np.ones(len(intervals), dtype=bool) if stage == 'full' else folds != stage
                trial, bins = np.where(observed & train[:, None])
                coordinates, inverse, weights = np.unique(np.c_[times[bins], intervals[trial]], axis=0,
                                                          return_inverse=True, return_counts=True)
                t, interval = coordinates.T
                grids = {False: coarse_grid(model, t, interval, weights)}
                if model == 'plain':
                    grids[True] = coarse_grid(model, t, interval, weights, True)
                test_trial, test_bins = np.where(observed & ~train[:, None])
                for index, cell in enumerate(cells.itertuples()):
                    mirrored = model == 'plain' and cell.group == 'tic_up'
                    y = np.bincount(inverse, weights=signals[index, trial, bins]) / weights
                    params, ok, nfev = fit_model(model, t, interval, y, weights, mirrored, grids[mirrored])
                    if not ok:
                        failures.append(dict(session=session, neu_id=int(cell.neu_id), model=model,
                                             stage=str(stage), issue='least_squares did not converge'))
                    clip_count, beyond_count = 0, 0
                    if model != 'plain':
                        peak, onset = peak_and_onset(intervals[train], params[2:], model.removeprefix('revised_'))
                        clip_count = int((params[2] > peak).sum())
                        beyond_count = int((peak > intervals[train]).sum())
                        clipping.append(dict(session=session, neu_id=int(cell.neu_id), group=cell.group,
                                             model=model, stage=str(stage), trials=int(train.sum()),
                                             clipped_trials=clip_count, peak_beyond_endpoint_trials=beyond_count))
                    if stage == 'full':
                        row = dict(session=session, neu_id=int(cell.neu_id), group=cell.group,
                                   model=model, fit_success=bool(ok), nfev=nfev,
                                   trials=len(intervals), clipped_trials=clip_count,
                                   peak_beyond_endpoint_trials=beyond_count)
                        row.update(zip(['a', 'B'] + KEYS[model], params))
                        row['B_at_lower'] = params[1] <= 1e-5
                        for key, value, lo, hi in zip(KEYS[model], params[2:], LOWER[model], UPPER[model]):
                            row[f'{key}_at_lower'] = bool(np.isclose(value, lo, rtol=0, atol=1e-5))
                            row[f'{key}_at_upper'] = bool(np.isclose(value, hi, rtol=0, atol=1e-5))
                        parameters.append(row)
                    else:
                        held[index, test_trial, test_bins] = predict_model(
                            model, times[test_bins], intervals[test_trial], params, mirrored)
                print(session, model, stage, 'seconds', round(time.perf_counter() - started, 1), flush=True)
            saved.create_dataset(model, data=held, compression='lzf')
            del grids
        with h5py.File(WIDER / f'{session}_wider_oof.h5', 'r') as original, h5py.File(
                FIRST / f'{session}_oof.h5', 'r') as first:
            np.testing.assert_array_equal(original['neu_id'][:], ids)
            np.testing.assert_array_equal(first['neu_id'][:], ids)
            np.testing.assert_array_equal(original['trial_rows'][:], data['trial_rows'] + 1)
            np.testing.assert_array_equal(original['t'][:], times)
            for index, cell in enumerate(cells.itertuples()):
                exponential = 'rise' if cell.group == 'tic_up' else 'decay'
                predictions = {name: saved[name][index] for name in MODELS}
                predictions.update({name: original[name][index] for name in ['gaussian', exponential]})
                predictions['first_bump'] = first[PRIMARY[cell.group]][index]
                for scoring in ['all', 'post_last_lick']:
                    signal = signals[index].astype(float).copy()
                    if scoring != 'all':
                        signal[:, times < 0] = np.nan
                    for prediction in predictions.values():
                        np.testing.assert_array_equal(np.isfinite(prediction), observed)
                    errors = {name: trial_mse(signal, fitted) for name, fitted in predictions.items()}
                    for a in MODELS:
                        for b in ['gaussian', exponential, 'first_bump']:
                            rows.append(dict(session=session, neu_id=int(cell.neu_id), group=cell.group,
                                             model_a=a, model_b=b, scoring=scoring,
                                             **contrast_row(errors[a], errors[b], variances.loc[cell.neu_id], bootstrap)))
    pd.DataFrame(parameters).to_csv(OUTPUT / f'{session}_parameters.csv', index=False)
    pd.DataFrame(rows).to_csv(OUTPUT / f'{session}_contrasts.csv', index=False)
    pd.DataFrame(clipping).to_csv(OUTPUT / f'{session}_clipping.csv', index=False)
    status = dict(session=session, seconds=time.perf_counter() - started, failures=failures, cells=len(ids))
    (OUTPUT / f'{session}_status.json').write_text(json.dumps(status, indent=2))
    print(session, 'finished', len(ids), 'cells;', len(failures), 'failures', flush=True)
    return status


#%% summaries
def summarise(seconds):
    parameters = pd.concat([pd.read_csv(p) for p in sorted(OUTPUT.glob('*_parameters.csv'))], ignore_index=True)
    cells = pd.concat([pd.read_csv(p) for p in sorted(OUTPUT.glob('*_contrasts.csv'))], ignore_index=True)
    clipping = pd.concat([pd.read_csv(p) for p in sorted(OUTPUT.glob('*_clipping.csv'))], ignore_index=True)
    parameters.to_csv(OUTPUT / 'parameters.csv', index=False)
    cells.to_csv(OUTPUT / 'contrasts.csv', index=False)
    clipping.to_csv(OUTPUT / 'clipping.csv', index=False)
    rows = []
    for (group, a, b, scoring), part in cells.groupby(['group', 'model_a', 'model_b', 'scoring']):
        medians = part.groupby('session').mean_contrast.median()
        rows.append(dict(group=group, model_a=a, model_b=b, scoring=scoring, n=len(part), sessions=len(medians),
                         better=100 * part.decision.eq('favour_a').mean(), worse=100 * part.decision.eq('favour_b').mean(),
                         undecided=100 * part.decision.eq('undecided').mean(),
                         median_session_contrast=medians.median(), wilcoxon_p=signed_test(medians)))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT / 'comparison_summary.csv', index=False)
    rows = []
    for (group, model), part in parameters.groupby(['group', 'model']):
        for key in ['a', 'B'] + KEYS[model]:
            rows.append(dict(group=group, model=model, parameter=key, n=len(part),
                             median=part[key].median(), q25=part[key].quantile(.25), q75=part[key].quantile(.75),
                             lower_bound_pct=100 * part[f'{key}_at_lower'].mean() if key != 'a' else np.nan,
                             upper_bound_pct=100 * part[f'{key}_at_upper'].mean() if key in KEYS[model] else np.nan))
    parameter_summary = pd.DataFrame(rows)
    parameter_summary.to_csv(OUTPUT / 'parameter_summary.csv', index=False)
    rows = []
    for (group, model, stage), part in clipping.groupby(['group', 'model', 'stage']):
        rows.append(dict(group=group, model=model, stage=stage, fits=len(part),
                         fits_clipped_pct=100 * (part.clipped_trials > 0).mean(),
                         clipped_trial_pct=100 * part.clipped_trials.sum() / part.trials.sum(),
                         peak_beyond_endpoint_trial_pct=100 * part.peak_beyond_endpoint_trials.sum() / part.trials.sum()))
    clip_summary = pd.DataFrame(rows)
    clip_summary.to_csv(OUTPUT / 'clipping_summary.csv', index=False)
    statuses = [json.loads(p.read_text()) for p in OUTPUT.glob('*_status.json')]
    failures = [f for status in statuses for f in status['failures']]
    pd.DataFrame(failures, columns=['session', 'neu_id', 'model', 'stage', 'issue']).to_csv(OUTPUT / 'failures.csv', index=False)
    text = '# revised bumps and plain exponentials\n\n'
    text += 'The original data, five folds, trial bootstrap resamples, flat residual variance, contrast sign and 95% decision interval are retained. All observed trial bins enter each fit, with identical (t, T) pairs averaged and weighted by count, without subsampling. All-bin scoring is primary; t >= 0 is scored from the same held-out predictions without refitting. Positive contrasts favour model_a. Wilcoxon p-values are raw two-sided tests of the 15 session medians against zero. The first bump always uses last anchoring for ramp-down TICs and time cells, and next anchoring for ramp-up TICs.\n\n'
    text += 'Both revised anchorings are fitted for every cell. A coarse grid (t0: -1, 0, 1, 2, 3 s; u: 0, 0.5, 1, 2, 3, 4, 5.5, 7 s; four logarithmic values per tau) profiles a and non-negative B, followed by bounded least squares with the profiled analytic Jacobian. The plain model uses 10 evenly spaced mu and 12 logarithmic tau starts, matching the existing exponential grid density. The plain model is exactly a + B exp(-max(x - mu, 0) / tau), with x = T - t for ramp-up TICs and x = t otherwise.\n\n'
    text += 'The first completed session (A215-20251013-02, 75 cells) was reported before the full run, then rerun unchanged with the other sessions; its initial timing is retained in first_session_run.json.\n\n'
    text += markdown(comparison) + '\n\n## full-data parameters\n\n' + markdown(parameter_summary)
    text += '\n\nThe baseline a is unbounded; B has lower bound zero and no upper bound. Nonlinear bound flags use absolute tolerance 1e-5. IQRs describe variation between cells. When B is zero, nonlinear shape parameters are unidentifiable; these fits remain in the requested summaries and are flagged at the amplitude lower bound.\n\n'
    text += 'For the revised last anchoring, the effective trial onset is min(t0, p); clipping is counted once per cell and trial, not once per observed bin. For next anchoring, p = max(T - u, t0) already guarantees t0 <= p; if t0 > T the peak is beyond the trial and all observed bins are baseline. Empty pieces are omitted, and coincident onset/peak or peak/endpoint values use the original baseline endpoint convention.\n\n'
    text += markdown(clip_summary) + '\n\n## fit failures\n\n' + markdown(pd.DataFrame(failures, columns=['session', 'neu_id', 'model', 'stage', 'issue']))
    text += f'\n\nFit failures: {len(failures)} of {len(parameters) * 6} fits. Fitting wall time: {seconds:.1f} s. Coarse-grid starts can still reach local minima; the grid and refinement are fixed before inspecting results.\n'
    (OUTPUT / 'summary.md').write_text(text, encoding='utf-8')
    (OUTPUT / 'run_report.json').write_text(json.dumps(dict(seconds=seconds, failures=len(failures), fits=len(parameters)*6), indent=2))
    print(text, flush=True)
    return comparison

#%% figures from the saved comparisons
def comparison_figure(table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from analysis.two_exponential_comparison import p_label

    plt.rcParams.update({'font.size': 16, 'axes.titlesize': 19, 'axes.labelsize': 16,
                         'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 14,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'pdf.use14corefonts': False})
    labels = ['Ramp-down TIC', 'Ramp-up TIC', 'Time cells']
    fig, axes = plt.subplots(1, 2, figsize=(13., 5.8), sharey=True)
    for ax, panel, reference, title in zip(axes, ['A', 'B'], ['exponential', 'gaussian'],
            ['Revised two-exponential vs exponential', 'Revised two-exponential vs Gaussian']):
        rows = []
        for group in GROUPS:
            model = 'revised_' + PRIMARY[group]
            other = ('rise' if group == 'tic_up' else 'decay') if reference == 'exponential' else 'gaussian'
            row = table[(table.group == group) & (table.model_a == model) &
                        (table.model_b == other) & (table.scoring == 'all')].iloc[0]
            rows.append(row)
        selected = pd.DataFrame(rows).reset_index(drop=True)
        bottom = np.zeros(3)
        for field, colour, label in zip(['better', 'worse', 'undecided'],
                ['#4D4D4D', '#E69F00', '#BDBDBD'], ['Revised two-exponential better', 'Other model better', 'No clear difference']):
            values = selected[field].to_numpy(float)
            ax.bar(np.arange(3), values, bottom=bottom, color=colour, width=.65, label=label)
            bottom += values
        np.testing.assert_allclose(bottom, 100.)
        for i, row in selected.iterrows():
            ax.text(i + .12 * (1 - i), 103., p_label(row.wilcoxon_p), ha='center', va='bottom', fontsize=15)
        ax.set(title=title, ylabel='Cells (%)', ylim=(0., 114.), yticks=np.arange(0, 101, 20),
               xticks=range(3), xticklabels=labels)
        ax.text(-.12, 1.05, panel, transform=ax.transAxes, fontsize=22, fontweight='bold', ha='right')
    axes[1].set_ylabel('')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .995), ncol=3,
               frameon=False, handlelength=1., columnspacing=1.)
    fig.subplots_adjust(left=.085, right=.98, bottom=.17, top=.78, wspace=.28)
    fig.savefig(FIGURES / 'model_comparison_v2.png', dpi=300)
    fig.savefig(FIGURES / 'model_comparison_v2.pdf')
    plt.close(fig)


#%% averaged-trace extension
def averaged_quality():
    from analysis.fit_quality import average_traces, r_squared

    started = time.perf_counter()
    parameters = pd.read_csv(OUTPUT / 'parameters.csv')
    first_parameters = pd.read_csv(FIRST / 'parameters.csv')
    original = pd.read_csv(ROOT / 'outputs' / 'fit_quality_two_exponential' / 'fit_quality_cells.csv')
    examples = pd.read_csv(ROOT / 'outputs' / 'fit_quality' / 'fit_quality_examples.csv')
    curves, rows = {}, []
    for path in sorted(WIDER.glob('*_wider_parameters.csv')):
        saved = pd.read_csv(path)
        session = saved.session.iloc[0]
        data = np.load(DATA / f'{session}.npz')
        signals, times, intervals, ids = [data[k] for k in ['dff', 't', 'T_k', 'neu_id']]
        indices = {int(neuron): i for i, neuron in enumerate(ids)}
        fitted = parameters[parameters.session == session].set_index(['neu_id', 'model'])
        old_bump = first_parameters[first_parameters.session == session].set_index(['neu_id', 'anchoring'])
        for cell in saved.itertuples():
            signal = signals[indices[int(cell.neu_id)]].astype(float)
            model = 'revised_' + PRIMARY[cell.group]
            row = fitted.loc[(cell.neu_id, model)]
            revised_ok = bool(row.fit_success)
            revised = predict_model(model, times[None, :], intervals[:, None],
                                    row[['a', 'B'] + KEYS[model]].to_numpy(float))
            row = fitted.loc[(cell.neu_id, 'plain')]
            plain_ok = bool(row.fit_success)
            plain = predict_model('plain', times[None, :], intervals[:, None],
                                  row[['a', 'B', 'mu', 'tau']].to_numpy(float), cell.group == 'tic_up')
            row = old_bump.loc[(cell.neu_id, PRIMARY[cell.group])]
            bump = predict_bump(times[None, :], intervals[:, None],
                                row[['a', 'B', 'u', 'tau1', 'tau2']].to_numpy(float), PRIMARY[cell.group])
            for alignment in ['last_lick', 'next_lick'] if cell.group == 'tic_up' else ['last_lick']:
                trace = average_traces(signal, times, intervals, cell, alignment, additional=[bump, revised, plain])
                record = dict(session=session, neu_id=int(cell.neu_id), group=cell.group, alignment=alignment,
                              revised_anchoring=PRIMARY[cell.group], revised_fit_success=revised_ok,
                              plain_fit_success=plain_ok)
                record.update({f'r2_{name}': r_squared(trace[1], values) for name, values in
                               zip(['gaussian', 'exponential', 'bump', 'revised', 'plain'], trace[2:])})
                rows.append(record)
                if ((examples.session == session) & (examples.neu_id == cell.neu_id) & (examples.alignment == alignment)).any():
                    curves[(session, int(cell.neu_id), alignment)] = trace
        print(session, 'new averaged traces complete', flush=True)
    cells = pd.DataFrame(rows)
    keys = ['session', 'neu_id', 'group', 'alignment']
    check = cells.merge(original, on=keys, validate='one_to_one', suffixes=('', '_original'))
    assert len(check) == len(cells) == len(original)
    for name in ['gaussian', 'exponential', 'bump']:
        np.testing.assert_allclose(check[f'r2_{name}'], check[f'r2_{name}_original'], atol=1e-12, rtol=0)
    cells.to_csv(OUTPUT / 'fit_quality_cells.csv', index=False)
    examples = examples[keys + ['percentile']].merge(cells, on=keys, validate='one_to_one')
    examples.to_csv(OUTPUT / 'fit_quality_examples.csv', index=False)
    rows = []
    for (group, alignment), part in cells.groupby(['group', 'alignment']):
        for model in ['gaussian', 'exponential', 'bump', 'revised', 'plain']:
            values = part[f'r2_{model}']
            rows.append(dict(group=group, alignment=alignment, model=model, n=len(part), defined=values.notna().sum(),
                             median=values.median(), q25=values.quantile(.25), q75=values.quantile(.75),
                             above_half_pct=100 * (values > .5).mean()))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / 'fit_quality_summary.csv', index=False)
    text = '# trial-averaged fit quality with the revised bump and plain exponential\n\n'
    text += 'The plain exponential remains in this table and is omitted from both figures; only the revised bump is added to the example curves and box plots. All curves use full-data parameters and the same observed trial-bin mask, alignment interpolation and minimum of 10 contributing trials as the original averaged-trace analysis. The original Gaussian, exponential and first-bump R2 values are unchanged, as are the 12 example IDs. The revised bump uses last anchoring for ramp-down TICs and time cells, and next anchoring for ramp-up TICs. Ramp-up traces run from negative time to zero relative to the next lick; both alignments remain in the table. These R2 values describe fitted averages, while the separate held-out contrasts test prediction.\n\n'
    text += markdown(summary) + '\n\n## unchanged example cells\n\n' + markdown(examples)
    make_examples_figure(cells, examples, curves, FIGURES / 'fit_examples_v2')
    seconds = time.perf_counter() - started
    text += f'\n\nAveraging and figure wall time: {seconds:.1f} s. Both figures use 13-inch-wide canvases and fonts of at least 14 pt; PDF fonts are embedded TrueType.\n'
    (OUTPUT / 'fit_quality_summary.md').write_text(text, encoding='utf-8')
    print(text, flush=True)
    return seconds


def make_examples_figure(cells, examples, curves, destination):
    from analysis.fit_quality import LABELS, COLOURS
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({'font.size': 16, 'axes.labelsize': 14, 'axes.titlesize': 17,
                         'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'pdf.use14corefonts': False})
    fig = plt.figure(figsize=(13., 15.5))
    grid = fig.add_gridspec(4, 1, height_ratios=[1., 1., 1., 1.15], hspace=.95,
                            left=.08, right=.985, bottom=.095, top=.89)
    bottom_grid = grid[3].subgridspec(1, 3, wspace=.65)
    for row, (group, label, colour) in enumerate(zip(GROUPS, LABELS, COLOURS)):
        selected = examples[examples.group.eq(group)].sort_values('percentile')
        example_grid = grid[row].subgridspec(1, 4, wspace=.75)
        for column, cell in enumerate(selected.itertuples()):
            axis = fig.add_subplot(example_grid[column])
            x, observed, gaussian, exponential, bump, revised, plain = curves[(cell.session, int(cell.neu_id), cell.alignment)]
            if group == 'tic_up':
                x = -x[::-1]
                observed, gaussian, exponential, bump, revised, plain = [v[::-1] for v in [observed, gaussian, exponential, bump, revised, plain]]
            axis.plot(x, observed, color=colour, linewidth=2.)
            axis.plot(x, gaussian, color='black', linewidth=1.5)
            axis.plot(x, exponential, color='black', linewidth=1.5, linestyle='--')
            axis.plot(x, bump, color='black', linewidth=1.8, linestyle=':')
            axis.plot(x, revised, color='#0072B2', linewidth=1.8, linestyle='-.')
            axis.set(xlabel='Time relative to\nnext lick (s)' if group == 'tic_up' else 'Time from\nlast lick (s)',
                     ylabel='dF/F')
            axis.yaxis.labelpad = 1.
            axis.xaxis.set_major_locator(MaxNLocator(4))
            axis.yaxis.set_major_locator(MaxNLocator(3))
            axis.spines[['top', 'right']].set_visible(False)
            if column == 0:
                fig.text(.08, axis.get_position().y1 + .015, label, fontsize=17, color=colour)
        alignment = 'next_lick' if group == 'tic_up' else 'last_lick'
        part = cells[cells.group.eq(group) & cells.alignment.eq(alignment)]
        axis = fig.add_subplot(bottom_grid[row])
        boxes = axis.boxplot([part[f'r2_{name}'].dropna() for name in ['gaussian', 'exponential', 'bump', 'revised']],
                             positions=[1., 2.5, 4., 5.5], widths=.55,
                             tick_labels=['Gaussian', 'Exponential', 'First bump', 'Revised bump'], patch_artist=True,
                             medianprops={'color': 'black', 'linewidth': 1.5},
                             flierprops={'markersize': 3, 'markeredgecolor': colour, 'alpha': .5})
        for box in boxes['boxes']:
            box.set(facecolor=colour, alpha=.55)
        axis.tick_params(axis='x', labelrotation=45)
        for text in axis.get_xticklabels():
            text.set_horizontalalignment('right')
        axis.axhline(0, color='#999999', linewidth=.8)
        axis.set(title=label, ylabel='Variance explained (R\u00b2)')
        axis.spines[['top', 'right']].set_visible(False)
    fig.legend(handles=[Line2D([0], [0], color='#777777', linewidth=2., label='Observed mean (group colour)'),
                        Line2D([0], [0], color='black', linewidth=1.5, label='Gaussian fit'),
                        Line2D([0], [0], color='black', linewidth=1.5, linestyle='--', label='Exponential fit'),
                        Line2D([0], [0], color='black', linewidth=1.8, linestyle=':', label='First two-exponential fit'),
                        Line2D([0], [0], color='#0072B2', linewidth=1.8, linestyle='-.', label='Revised two-exponential fit')],
               loc='upper center', bbox_to_anchor=(.5, .985), frameon=False, ncol=2)
    fig.savefig(destination.with_suffix('.png'), dpi=300)
    fig.savefig(destination.with_suffix('.pdf'))
    plt.close(fig)


#%% main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session')
    parser.add_argument('--summarise-only', action='store_true')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.session:
        session_job(args.session)
        return
    if not args.summarise_only:
        sessions = sorted(p.name.replace('_wider_parameters.csv', '') for p in WIDER.glob('*_wider_parameters.csv'))
        with ProcessPoolExecutor(max_workers=6) as pool:
            list(pool.map(session_job, sessions))
        comparison = summarise(time.perf_counter() - started)
    else:
        comparison = pd.read_csv(OUTPUT / 'comparison_summary.csv')
    comparison_figure(comparison)
    averaging_seconds = averaged_quality()
    report = json.loads((OUTPUT / 'run_report.json').read_text())
    report['averaging_seconds'] = averaging_seconds
    if (OUTPUT / 'first_session_run.json').exists():
        report['completed_session_check_seconds'] = json.loads((OUTPUT / 'first_session_run.json').read_text())['seconds']
    report['figure_plain_exponential_omitted'] = True
    (OUTPUT / 'run_report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
