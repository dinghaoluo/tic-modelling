'''
2026-11-07
audit wider-window model evidence without changing the fitted models
@author: Dinghao Luo
'''

#%% imports
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'binned_v1_3'
FITS = ROOT / 'outputs' / 'fits_v1_2'
WIDER = ROOT / 'outputs' / 'undecided_check'
OUTPUT = ROOT / 'outputs' / 'wider_evidence'
FIGURES = ROOT / 'figures' / 'wider_evidence'
SEED = 20261107


def trial_mse(observed, prediction):
    residual = observed - prediction
    return np.nanmean(residual ** 2, axis=1)


def trial_contrast(error_a, error_b, variance):
    return (error_b - error_a) / (2. * variance)


def bootstrap_support(values, bootstrap):
    means = values[bootstrap].mean(axis=1)
    return dict(mean=float(values.mean()), lower=float(np.percentile(means, 2.5)),
                upper=float(np.percentile(means, 97.5)),
                exp_support=float(np.mean(means < 0.)),
                gaussian_support=float(np.mean(means > 0.)))


def audit_session(session, rng):
    data = np.load(DATA / f'{session}.npz')
    folds = np.load(FITS / f'{session}_folds.npz')
    contrasts = pd.read_csv(WIDER / f'{session}_wider_contrasts.csv')
    parameters = pd.read_csv(WIDER / f'{session}_wider_parameters.csv')
    with h5py.File(WIDER / f'{session}_wider_oof.h5', 'r') as saved:
        np.testing.assert_array_equal(saved['trial_rows'][:], data['trial_rows'] + 1)
        np.testing.assert_array_equal(saved['neu_id'][:], parameters.neu_id.to_numpy())
        np.testing.assert_array_equal(folds['trial_rows'], data['trial_rows'])
        signal = data['dff']
        rows = []
        for neuron, row in parameters.reset_index(drop=True).iterrows():
            observed = signal[neuron]
            finite = np.isfinite(observed)
            errors = {}
            for model in ['gaussian', 'decay', 'rise', 'flat']:
                prediction = saved[model][neuron]
                np.testing.assert_array_equal(np.isfinite(prediction), finite)
                errors[model] = trial_mse(observed, prediction)
            variance = float(contrasts.loc[(contrasts.neu_id == row.neu_id) &
                                           (contrasts.contrast == ('gaussian_minus_rise' if row.group == 'tic_up'
                                                                    else 'gaussian_minus_decay')), 'flat_residual_variance'].iloc[0])
            expected = 'rise' if row.group == 'tic_up' else 'decay'
            difference = trial_contrast(errors['gaussian'], errors[expected], variance)
            saved_row = contrasts[(contrasts.neu_id == row.neu_id) &
                                  (contrasts.contrast == f'gaussian_minus_{expected}')].iloc[0]
            np.testing.assert_allclose(difference.mean(), saved_row.mean_contrast, rtol=0, atol=1e-8)
            n_trials = len(difference)
            bootstrap = folds['bootstrap_trials']
            support = bootstrap_support(difference, bootstrap)
            for sample_fraction in [.25, .5, .75, 1.]:
                sample_size = max(10, int(round(sample_fraction * n_trials)))
                draws = rng.integers(0, n_trials, size=(500, sample_size))
                means = difference[draws].mean(axis=1)
                ci_low = np.percentile(means, 2.5)
                ci_high = np.percentile(means, 97.5)
                rows.append(dict(session=session, neu_id=int(row.neu_id), group=row.group,
                                 n_trials=n_trials, sample_fraction=sample_fraction, sample_size=sample_size,
                                 mean_contrast=support['mean'], lower=support['lower'], upper=support['upper'],
                                 exp_support=support['exp_support'], gaussian_support=support['gaussian_support'],
                                 undecided=bool(support['lower'] <= 0 <= support['upper']),
                                 sample_exp_support=float(np.mean(means < 0)),
                                 sample_ci_excludes_zero=float((ci_low > 0) | (ci_high < 0))))
        return rows


def summary(frame):
    rows = []
    full = frame[frame.sample_fraction == 1.]
    for group, part in full.groupby('group'):
        rows.append(dict(group=group, n_cells=len(part), n_sessions=part.session.nunique(),
                         median_contrast=part.mean_contrast.median(),
                         median_exp_support=part.exp_support.median(),
                         pct_exp_support_ge_95=100 * (part.exp_support >= .95).mean(),
                         pct_gaussian_support_ge_95=100 * (part.gaussian_support >= .95).mean(),
                         pct_undecided=100 * part.undecided.mean(),
                         median_n_trials=part.n_trials.median()))
    calibration = []
    for (group, sample_fraction), part in frame.groupby(['group', 'sample_fraction']):
        calibration.append(dict(group=group, sample_fraction=float(sample_fraction), n_cells=len(part),
                                median_sample_exp_support=100 * part.sample_exp_support.median(),
                                pct_cells_ci_excludes_zero=100 * part.sample_ci_excludes_zero.mean()))
    return pd.DataFrame(rows), pd.DataFrame(calibration)


def make_figure(full, calibration):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'pdf.use14corefonts': False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colours = {'tic_down': '#4477AA', 'tic_up': '#CC6677', 'time_cell': '#228833'}
    for group, part in full.groupby('group'):
        axes[0].hist(part.exp_support, bins=np.linspace(0, 1, 21), density=True, alpha=.5,
                     color=colours[group], label=group.replace('_', ' '))
    axes[0].set(xlabel='bootstrap support for exponential', ylabel='density')
    axes[0].legend(frameon=False, fontsize=8)
    for group, part in calibration.groupby('group'):
        axes[1].plot(part.sample_fraction, part.pct_cells_ci_excludes_zero, 'o-',
                     color=colours[group], label=group.replace('_', ' '))
    axes[1].set(xlabel='fraction of available trials', ylabel='% cells with CI excluding zero')
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    fig.suptitle('wider-window held-out evidence audit')
    fig.tight_layout()
    fig.savefig(FIGURES / 'evidence_audit.png', dpi=300)
    fig.savefig(FIGURES / 'evidence_audit.pdf')
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    sessions = sorted(path.name.replace('_wider_contrasts.csv', '')
                      for path in WIDER.glob('*_wider_contrasts.csv'))
    rng = np.random.default_rng(SEED)
    rows = [row for session in sessions for row in audit_session(session, rng)]
    frame = pd.DataFrame(rows)
    full, calibration = summary(frame)
    frame.to_csv(OUTPUT / 'cell_evidence.csv', index=False)
    full.to_csv(OUTPUT / 'group_evidence.csv', index=False)
    calibration.to_csv(OUTPUT / 'trial_calibration.csv', index=False)
    report = ['# wider-window evidence audit', '',
              'Per-trial Gaussian-minus-exponential log-likelihood contrasts were reconstructed from the wider-window observations and out-of-fold predictions. The saved wider contrast means were checked cell by cell before the bootstrap support and trial-count calibration were calculated.', '',
              '## group evidence', full.to_string(index=False), '',
              '## trial calibration', calibration.to_string(index=False), '',
              'The calibration resamples saved per-trial contrasts while keeping the fitted predictions fixed. It is a precision diagnostic, not a new fit; it shows whether the current trial counts are sufficient to resolve the existing model contrast.', '']
    (OUTPUT / 'summary.md').write_text('\n'.join(report), encoding='utf-8')
    make_figure(frame[frame.sample_size == frame.n_trials], calibration)
    print(full.to_string(index=False))
    print(calibration.to_string(index=False))


if __name__ == '__main__':
    main()
