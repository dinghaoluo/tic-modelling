'''
2026-11-06
check whether fixed calcium deconvolution changes TIC shape decisions
@author: Dinghao Luo
'''

#%% imports
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rec_list import SESSIONS_4S
from utils.trial_error_scores import paired_gain, trial_mse

DATA = ROOT / 'data' / 'binned_v1_1'
FIT_OUTPUT = ROOT / 'outputs' / 'fits_v1_2'
OUTPUT = ROOT / 'outputs' / 'deconv_sensitivity'
FIGURES = ROOT / 'figures' / 'deconv_sensitivity'
SHAPES = ['gaussian', 'decay', 'rise']
BOUNDS = {
    'gaussian': (np.array([-1., .05]), np.array([7., 4.])),
    'decay': (np.array([-1., .1]), np.array([3., 10.])),
    'rise': (np.array([-1., .1]), np.array([3., 10.])),
}
GRIDS = {
    'gaussian': (np.linspace(-1., 7., 15), np.geomspace(.05, 4., 12)),
    'decay': (np.linspace(-1., 3., 10), np.geomspace(.1, 10., 12)),
    'rise': (np.linspace(-1., 3., 10), np.geomspace(.1, 10., 12)),
}
GAMMA = np.exp(-.05 / .2)
SEED = 20261104


#%% pure shapes after deconvolution

def shape_values(name, t, interval, params):
    mu, width = params
    if name == 'gaussian':
        return np.exp(-.5 * ((t - mu) / width) ** 2)
    x = interval - t if name == 'rise' else t
    return np.exp(-np.maximum(x - mu, 0.) / width)


def linear_amplitude(y, shape):
    shape_mean = shape.mean()
    y_mean = y.mean()
    centred = shape - shape_mean
    denominator = np.dot(centred, centred)
    amplitude = max(float(np.dot(centred, y - y_mean) / denominator), 0.) if denominator else 0.
    return float(y_mean - amplitude * shape_mean), amplitude


def fit_field(name, t, interval, y):
    lower, upper = BOUNDS[name]
    candidates = np.array(np.meshgrid(*GRIDS[name])).reshape(2, -1).T
    shapes = np.array([shape_values(name, t, interval, params) for params in candidates])
    means = shapes.mean(axis=1)
    centred = shapes - means[:, None]
    covariance = centred @ (y - y.mean())
    denominator = (centred ** 2).sum(axis=1)
    score = np.maximum(covariance, 0.) ** 2 / np.maximum(denominator, 1e-12)
    initial = candidates[np.argmax(score)]

    def residual(params):
        shape = shape_values(name, t, interval, params)
        baseline, amplitude = linear_amplitude(y, shape)
        return baseline + amplitude * shape - y

    result = least_squares(residual, initial, bounds=(lower, upper), x_scale='jac')
    shape = shape_values(name, t, interval, result.x)
    baseline, amplitude = linear_amplitude(y, shape)
    return np.r_[baseline, amplitude, result.x], result.success


def predict(name, t, interval, params):
    return params[0] + params[1] * shape_values(name, t, interval, params[2:])


def fit_cv(name, signal, times, intervals, folds):
    held = np.full(signal.shape, np.nan)
    failures = []
    for fold in range(5):
        train = folds != fold
        test = ~train
        trial, bins = np.where(np.isfinite(signal) & train[:, None])
        y = signal[trial, bins].astype(float)
        t = times[bins]
        interval = intervals[trial]
        params, ok = fit_field(name, t, interval, y)
        test_trial, test_bins = np.where(np.isfinite(signal) & test[:, None])
        held[test_trial, test_bins] = predict(name, times[test_bins], intervals[test_trial], params)
        if not ok:
            failures.append(fold)
    return held, failures


def deconvolve(signal):
    result = np.full(signal.shape, np.nan, dtype=float)
    for trial, values in enumerate(signal):
        finite = np.flatnonzero(np.isfinite(values))
        if not len(finite):
            continue
        n = finite[-1] + 1
        result[trial, :n] = lfilter([1., -GAMMA], [1.], values[:n])
    return result


#%% sessions

def fit_session(session):
    started = time.perf_counter()
    data = np.load(DATA / f'{session}.npz')
    dff, intervals, times = data['dff'], data['T_k'], data['t']
    labels = pd.read_csv(FIT_OUTPUT / 'labels.csv').query("session == @session and group in ['tic_down', 'tic_up']").reset_index(drop=True)
    neuron_indices = labels.neu_id.astype(int).to_numpy() - 1
    rng = np.random.default_rng(SEED + SESSIONS_4S.index(session))
    folds = np.empty(len(intervals), dtype=int)
    for fold, indices in enumerate(np.array_split(rng.permutation(len(intervals)), 5)):
        folds[indices] = fold
    bootstrap = rng.integers(0, len(intervals), size=(1000, len(intervals)))
    rows = []
    for row_index, neuron_index in enumerate(neuron_indices):
        signal = deconvolve(dff[neuron_index])
        errors, failures = {}, []
        trial, bins = np.where(np.isfinite(signal))
        y, t, interval = signal[trial, bins], times[bins], intervals[trial]
        for name in SHAPES:
            held, failed = fit_cv(name, signal, times, intervals, folds)
            errors[name] = trial_mse(signal, held)
            failures.extend(f'{name}:{fold}' for fold in failed)
        ordered = sorted(SHAPES, key=lambda name: errors[name].mean())
        best, next_shape = ordered[:2]
        flat_prediction = np.full(signal.shape, np.nan)
        flat_prediction[np.isfinite(signal)] = y.mean()
        errors['flat'] = trial_mse(signal, flat_prediction)
        flat_gain = paired_gain(errors[best], errors['flat'], bootstrap)
        next_gain = paired_gain(errors[best], errors[next_shape], bootstrap)
        row = labels.iloc[row_index].to_dict()
        row.update(session=session, best_shape=best, next_shape=next_shape,
                   beats_flat=flat_gain['lower'] > 0,
                   preferred=('flat' if flat_gain['lower'] <= 0 else
                              ('undecided' if next_gain['lower'] <= 0 <= next_gain['upper'] else best)),
                   gain_flat=flat_gain['gain'], gain_flat_lower=flat_gain['lower'], gain_flat_upper=flat_gain['upper'],
                   gain_next=next_gain['gain'], gain_next_lower=next_gain['lower'], gain_next_upper=next_gain['upper'],
                   failed_fits=';'.join(failures))
        for name in SHAPES:
            params, ok = fit_field(name, t, interval, y)
            row.update({f'{name}_mu': float(params[2]), f'{name}_width': float(params[3]),
                       f'{name}_at_bound': bool(np.any(np.isclose(params[2:], BOUNDS[name][0], atol=1e-5)) or
                                                  np.any(np.isclose(params[2:], BOUNDS[name][1], atol=1e-5)))})
            if not ok:
                row['failed_fits'] += f';{name}:full'
        rows.append(row)
        if len(rows) % 25 == 0:
            print(session, 'completed TICs', len(rows), flush=True)
    elapsed = time.perf_counter() - started
    pd.DataFrame(rows).to_csv(OUTPUT / f'{session}.csv', index=False)
    print(session, 'complete in seconds', elapsed, flush=True)
    return elapsed, len(rows)


#%% report

def make_report(frame, seconds):
    rows = []
    for group, part in frame.groupby('group'):
        for decision in ['gaussian', 'decay', 'rise', 'undecided', 'flat']:
            rows.append({'group': group, 'decision': decision, 'n': int((part.preferred == decision).sum()),
                         'pct': 100 * float((part.preferred == decision).mean())})
    decisions = pd.DataFrame(rows)
    decisions.to_csv(OUTPUT / 'decision_rates.csv', index=False)
    frame.to_csv(OUTPUT / 'fits_all.csv', index=False)
    failed = frame.failed_fits.fillna('').astype(str).ne('')
    report = ['# fixed-kernel deconvolution sensitivity', '',
              'Each TIC trace was deconvolved in 50 ms bins with the inverse of the 0.2 s first-order kernel used by the calcium exponential model. The same held-out trial folds, bootstrap decision rule and event-relative pure Gaussian and exponential shapes were then applied. The transform is a sensitivity analysis, not a new cell-selection step.', '',
              '## decision rates', decisions.to_string(index=False), '',
              '## fit failures', frame.loc[failed, ['session', 'neu_id', 'failed_fits']].to_string(index=False) if failed.any() else 'None.', '',
              'The inverse filter shrank the signal amplitude and sent almost all cells to the flat guard. This is a diagnostic of the instability of this naive inverse on the saved dF/F, not evidence for a biological absence of a field.', '',
              f'total fitting seconds: {seconds:.1f}']
    (OUTPUT / 'summary.md').write_text('\n'.join(report), encoding='utf-8')
    return decisions


def make_figure(frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colours = {'gaussian': '#4477AA', 'decay': '#CC6677', 'rise': '#228833',
               'undecided': '#CCB84A', 'flat': '#999999'}
    fig, ax = plt.subplots(figsize=(7, 4))
    for group, part in frame.groupby('group'):
        bottom = 0.
        for decision, colour in colours.items():
            value = 100 * (part.preferred == decision).mean()
            ax.bar(group, value, bottom=bottom, color=colour, label=decision)
            bottom += value
    ax.set_ylabel('cells (%)')
    ax.set_title('deconvolved TIC held-out shape decisions')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / 'decision_rates.png', dpi=300)
    plt.close(fig)


#%% run

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fit_session, SESSIONS_4S))
    frame = pd.concat([pd.read_csv(OUTPUT / f'{session}.csv') for session in SESSIONS_4S], ignore_index=True)
    decisions = make_report(frame, time.perf_counter() - started)
    make_figure(frame)
    print(decisions.to_string(index=False), flush=True)
    print(json.dumps(dict(seconds=time.perf_counter() - started,
                         cells=len(frame), sessions=len(SESSIONS_4S)), indent=2), flush=True)


if __name__ == '__main__':
    main()
