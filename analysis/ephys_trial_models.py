'''
2026-11-06
fit held-out receptive-field models to ephys trials
@author: Dinghao Luo
'''

#%% imports
import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import least_squares
from scipy.special import log_ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.trial_error_scores import paired_gain, trial_mse

RECORDING_LIST = Path(r'C:\Users\luod\Downloads\Recording_List_imm.m')
TIC_TABLE = Path(r'C:\Users\luod\Downloads\TICnoInt.mat')
CLUSTER_FILE = Path(r'Z:\Kori\immobile_code\Ephys\Oct2026\PAC_Kmeans_noPCA_start_2\KmeansResults.mat')
OUTPUT = ROOT / 'outputs' / 'ephys_trial_models'
FIGURES = ROOT / 'figures' / 'ephys_trial_models'
KERNEL = False
SHAPES = ['gaussian', 'decay', 'rise']
BOUNDS = {
    'gaussian': (np.array([0., .05]), np.array([7., 4.])),
    'decay': (np.array([0., .1]), np.array([3., 10.])),
    'rise': (np.array([0., .1]), np.array([3., 10.])),
}
GRIDS = {
    'gaussian': (np.linspace(0., 7., 15), np.geomspace(.05, 4., 12)),
    'decay': (np.linspace(0., 3., 10), np.geomspace(.1, 10., 12)),
    'rise': (np.linspace(0., 3., 10), np.geomspace(.1, 10., 12)),
}
BIN_END = np.arange(1, 141) * .05
BIN_TIME = BIN_END - .026
SEED = 20261106


#%% inputs

def recording_paths():
    paths = []
    for line in RECORDING_LIST.read_text().splitlines():
        match = re.search(r"'([^']*Raphael_tests\\mice_expdata[^']*)'", line)
        if match:
            paths.append(Path(match.group(1)))
    if len(paths) != 15:
        raise ValueError(f'expected 15 ephys recording paths, found {len(paths)}')
    return paths


def label_table():
    import matio

    table = matio.load_from_mat(str(TIC_TABLE))['RiseLLDownLickNeuronID'].copy()
    clusters = loadmat(CLUSTER_FILE, squeeze_me=True, struct_as_record=False)['clusters']
    idx = np.asarray(clusters.idx).astype(int).ravel()
    x = np.asarray(clusters.X, dtype=float)
    if len(idx) != len(table) or len(idx) != len(x):
        raise ValueError('ephys TIC labels and cluster profiles do not match')
    centre = (x * np.arange(x.shape[1])).sum(axis=1) / x.sum(axis=1)
    means = {int(k): float(centre[idx == k].mean()) for k in np.unique(idx)}
    early_cluster = min(means, key=means.get)
    table['cluster_number'] = idx
    table['group'] = np.where(idx == early_cluster, 'tic_down', 'tic_up')
    return table, means


def source_files(folder):
    conv = next(folder.glob('*_convSpikesAligned_msess1_BefLastLick0.mat'))
    info = next(folder.glob('*_DataStructure_mazeSection1_TrialType1_Info.mat'))
    return conv, info, folder / 'FirstLick_LL.mat'


#%% spike-rate models

def shape_values(name, t, interval, params):
    mu, width = params
    if name == 'gaussian':
        return np.exp(-.5 * ((t - mu) / width) ** 2)
    x = interval - t if name == 'rise' else t
    if KERNEL:
        z = (x - mu) / .2
        return np.exp(.5 * (.2 / width) ** 2 - (x - mu) / width + log_ndtr(z - .2 / width))
    return np.exp(-np.maximum(x - mu, 0.) / width)


def linear_amplitude(y, shape):
    shape_mean = shape.mean()
    y_mean = y.mean()
    centred = shape - shape_mean
    denominator = np.dot(centred, centred)
    amplitude = max(float(np.dot(centred, y - y_mean) / denominator), 0.) if denominator else 0.
    return float(y_mean - amplitude * shape_mean), amplitude


def fit_field(name, t, interval, y):
    if name == 'flat':
        return np.array([float(y.mean())]), True
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
    if name == 'flat':
        return np.full_like(t, params[0], dtype=float)
    return params[0] + params[1] * shape_values(name, t, interval, params[2:])


def fit_cv(name, signal, times, intervals, folds):
    held = np.full(signal.shape, np.nan)
    failures = []
    for fold in range(5):
        train = folds != fold
        test = ~train
        trial, bins = np.where(np.isfinite(signal) & train[:, None])
        y = signal[trial, bins].astype(float)
        t = np.broadcast_to(times[bins], y.shape)
        interval = np.broadcast_to(intervals[trial], y.shape)
        params, ok = fit_field(name, t, interval, y)
        test_trial, test_bins = np.where(np.isfinite(signal) & test[:, None])
        prediction = predict(name, np.broadcast_to(times[test_bins], test_bins.shape),
                             np.broadcast_to(intervals[test_trial], test_trial.shape), params)
        held[test_trial, test_bins] = prediction
        if not ok:
            failures.append(fold)
    return held, failures


#%% data loading

def load_session(folder, labels):
    conv_path, info_path, first_path = source_files(folder)
    first = np.asarray(loadmat(first_path, squeeze_me=True)['FirstLick'], dtype=float).ravel()
    beh = loadmat(info_path, squeeze_me=True, struct_as_record=False)['beh']
    all_trials = np.asarray(beh.indTrCtrl, dtype=int).ravel() - 1
    all_trials = all_trials[1:]
    valid = all_trials[(first[all_trials] >= 1.) & (first[all_trials] <= 6.998)]
    intervals = first[valid]

    with h5py.File(conv_path, 'r') as source:
        time = np.asarray(source['timeStepRun']).ravel() / 1250.
        refs = source['filteredSpikeArrayLastLickOnSet']
        rows = labels[labels.rec_name == folder.name].copy()
        signals = []
        for neu_id in rows.neu_id.astype(int):
            raw = np.asarray(source[refs[neu_id - 1, 0]][:], dtype=float)
            raw = raw[:, valid]
            binned = np.empty((len(valid), len(BIN_TIME)), dtype=float)
            for index, end in enumerate(BIN_END):
                selected = (time >= end - .05) & (time < end)
                binned[:, index] = raw[selected].mean(axis=0)
            binned[BIN_END[None, :] > intervals[:, None] + 1e-12] = np.nan
            signals.append(binned)

    return folder.name, rows.reset_index(drop=True), np.asarray(signals), intervals


def fit_job(job):
    return fit_session(*job)

def fit_session(folder, labels):
    session, rows, signals, intervals = load_session(folder, labels)
    rng = np.random.default_rng(SEED + int(sum(map(ord, session))))
    folds = np.empty(len(intervals), dtype=int)
    for fold, indices in enumerate(np.array_split(rng.permutation(len(intervals)), 5)):
        folds[indices] = fold
    bootstrap = rng.integers(0, len(intervals), size=(1000, len(intervals)))
    output = []
    for neuron, signal in enumerate(signals):
        row = rows.iloc[neuron].to_dict()
        row['session'] = session
        errors = {}
        predictions = {}
        failures = []
        trial, bins = np.where(np.isfinite(signal))
        y = signal[trial, bins]
        t = BIN_TIME[bins]
        interval = intervals[trial]
        for name in ['flat', *SHAPES]:
            held, failed = fit_cv(name, signal, BIN_TIME, intervals, folds)
            predictions[name] = held
            errors[name] = trial_mse(signal, held)
            failures.extend(f'{name}:{fold}' for fold in failed)
            params, ok = fit_field(name, t, interval, y)
            if name != 'flat':
                row.update({f'{name}_mu': float(params[2]), f'{name}_width': float(params[3]),
                            f'{name}_at_bound': bool(np.isclose(params[2], BOUNDS[name][0][0]) or
                                                      np.isclose(params[2], BOUNDS[name][1][0]) or
                                                      np.isclose(params[3], BOUNDS[name][0][1]) or
                                                      np.isclose(params[3], BOUNDS[name][1][1]))})
            row[f'{name}_held_mse'] = float(errors[name].mean())
        ordered = sorted(SHAPES, key=lambda name: row[f'{name}_held_mse'])
        best, next_shape = ordered[:2]
        flat_gain = paired_gain(errors[best], errors['flat'], bootstrap)
        next_gain = paired_gain(errors[best], errors[next_shape], bootstrap)
        row.update(best_shape=best, next_shape=next_shape,
                   beats_flat=flat_gain['lower'] > 0,
                   preferred=('flat' if flat_gain['lower'] <= 0 else
                              ('undecided' if next_gain['lower'] <= 0 <= next_gain['upper'] else best)),
                   gain_flat=flat_gain['gain'], gain_flat_lower=flat_gain['lower'], gain_flat_upper=flat_gain['upper'],
                   gain_next=next_gain['gain'], gain_next_lower=next_gain['lower'], gain_next_upper=next_gain['upper'],
                   failed_fits=';'.join(failures))
        output.append(row)
    return output


#%% report

def make_report(frame, means):
    rows = []
    for group, part in frame.groupby('group'):
        for decision in ['gaussian', 'decay', 'rise', 'undecided', 'flat']:
            rows.append({'group': group, 'decision': decision,
                         'n': int((part.preferred == decision).sum()),
                         'pct': 100 * float((part.preferred == decision).mean())})
    decisions = pd.DataFrame(rows)
    decisions.to_csv(OUTPUT / 'decision_rates.csv', index=False)
    frame.to_csv(OUTPUT / 'fits_all.csv', index=False)
    report = '\n'.join([
        '# ephys trial-level shape comparison',
        '',
        'The ephys traces were binned to the same 50 ms interval grid as the calcium analysis. The original trial selection and first-lick mask were retained. The held-out trial scoring and bootstrap decision rule are unchanged; the output folder records whether the exponential kernel was enabled.',
        '',
        '## cluster centre of mass',
        pd.DataFrame([{'cluster_number': key, 'mean_profile_centre': value} for key, value in means.items()]).to_string(index=False),
        '',
        '## decision rates',
        decisions.to_string(index=False),
        '',
        '## failures',
        frame.loc[frame.failed_fits != '', ['session', 'neu_id', 'failed_fits']].to_string(index=False),
    ])
    (OUTPUT / 'summary.md').write_text(report, encoding='utf-8')
    return decisions


def make_figure(frame, decisions):
    calcium = pd.read_csv(ROOT / 'outputs' / 'fits_v1_2' / 'fits_all.csv')
    calcium = calcium[calcium.group.isin(['tic_down', 'tic_up'])].copy()
    calcium['modality'] = 'calcium'
    calcium['group'] = calcium.group.map({'tic_down': 'ramp-down', 'tic_up': 'ramp-up'})
    ephys = frame.copy()
    ephys['modality'] = 'ephys'
    ephys['group'] = ephys.group.map({'tic_down': 'ramp-down', 'tic_up': 'ramp-up'})
    combined = pd.concat([calcium[['modality', 'group', 'preferred']], ephys[['modality', 'group', 'preferred']]])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for position, (modality, part) in enumerate(combined.groupby('modality', sort=False)):
        for group in ['ramp-down', 'ramp-up']:
            selected = part[part.group == group]
            bottom = 0.
            for decision, colour in [('gaussian', '#4477AA'), ('decay', '#CC6677'), ('rise', '#228833'),
                                     ('undecided', '#CCB84A'), ('flat', '#999999')]:
                value = 100 * (selected.preferred == decision).mean()
                label = decision if position == 1 and group == 'ramp-down' else '_nolegend_'
                axes[position].bar(group, value, bottom=bottom, color=colour, label=label)
                bottom += value
        axes[position].set_title(modality)
        axes[position].set_ylabel('cells (%)')
        axes[position].spines['top'].set_visible(False)
        axes[position].spines['right'].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle('ephys and calcium held-out shape decisions')
    fig.tight_layout()
    fig.savefig(FIGURES / 'ephys_calcium_decisions.png', dpi=300)
    plt.close(fig)


#%% run

def init_worker(kernel):
    global KERNEL
    KERNEL = kernel

def main():
    global OUTPUT, FIGURES, KERNEL
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', action='store_true')
    args = parser.parse_args()
    KERNEL = args.kernel
    suffix = '_kernel' if KERNEL else ''
    OUTPUT = ROOT / 'outputs' / f'ephys_trial_models{suffix}'
    FIGURES = ROOT / 'figures' / f'ephys_trial_models{suffix}'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    labels, means = label_table()
    paths = recording_paths()
    jobs = [(path, labels) for path in paths]
    rows = []
    with ProcessPoolExecutor(max_workers=min(12, len(jobs)),
                             initializer=init_worker, initargs=(KERNEL,)) as executor:
        for result in executor.map(fit_job, jobs):
            rows.extend(result)
    frame = pd.DataFrame(rows)
    decisions = make_report(frame, means)
    make_figure(frame, decisions)


if __name__ == '__main__':
    main()