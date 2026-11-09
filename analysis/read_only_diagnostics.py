'''
2026-11-09
run read-only peak, alpha, width and trial-count diagnostics
@author: Dinghao Luo
'''

#%% imports
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import minimize_scalar
from scipy.signal import firwin, kaiser_beta
from scipy.stats import wilcoxon


#%% constants
GROUPS = ['tic_down', 'tic_up', 'tic_and_time_cell', 'time_cell']
GROUP_LABELS = {
    'tic_down': 'Ramp-down TIC',
    'tic_up': 'Ramp-up TIC',
    'tic_and_time_cell': 'TIC/time-cell overlap',
    'time_cell': 'Time cells',
}
COLOURS = {
    'tic_down': '#228833',
    'tic_up': '#4477AA',
    'tic_and_time_cell': '#999999',
    'time_cell': '#9467BD',
}


#%% loading
def load_vector(path):
    return np.asarray(loadmat(path, squeeze_me=True)['Peaks']).ravel().astype(float)


def load_profiles(path):
    return np.asarray(loadmat(
        path,
        variable_names=['avgFR_profile'],
        squeeze_me=True,
        verify_compressed_data_integrity=False,
    )['avgFR_profile'], dtype=float)


def load_wider_parameters(root):
    wide = root / 'outputs' / 'undecided_check'
    return pd.concat([pd.read_csv(path) for path in sorted(wide.glob('*_wider_parameters.csv'))], ignore_index=True)


def load_session_order(source_path):
    order = []
    for line in source_path.read_text(encoding='utf-8').splitlines():
        stripped = line.lstrip()
        if stripped.startswith('%') or '{' not in line:
            continue
        match = re.search(r'A\d{3}-\d{8}-\d{2}', line)
        if match:
            order.append(match.group(0))
    if len(order) != len(set(order)):
        raise ValueError('source session order contains duplicates')
    return order


def load_first_lick_medians(path, session_order):
    values = np.asarray(loadmat(path, squeeze_me=True)['Med3545']).ravel().astype(float)
    if len(values) != len(session_order):
        raise ValueError('first-lick session count does not match the source session order')
    return dict(zip(session_order, values))


#%% peak diagnostics
def peak_table(widths, field_peaks, tic_peaks):
    field = widths[(widths.subset == 'all') & widths.neuron_type.eq('time_cell')].copy()
    tic = widths[
        (widths.subset == 'all') &
        widths.neuron_type.isin(['tic_down', 'tic_up', 'tic_and_time_cell'])
    ].copy()
    if len(field) != len(field_peaks) or len(tic) != len(tic_peaks):
        raise ValueError('peak arrays do not match the saved width row counts')
    field['threshold_peak_s'] = (field_peaks - 1.) * .002
    tic['threshold_peak_s'] = (tic_peaks - 1.) * .002
    return pd.concat([
        field[['session', 'neu_id', 'neuron_type', 'threshold_peak_s']],
        tic[['session', 'neu_id', 'neuron_type', 'threshold_peak_s']],
    ], ignore_index=True)


def peak_fraction_table(peaks, first_lick):
    rows = []
    for group in GROUPS:
        part = peaks[peaks.neuron_type.eq(group)].copy()
        part['session_first_lick_s'] = part.session.map(first_lick)
        rows.append({
            'group': GROUP_LABELS[group],
            'n_cells': len(part),
            'n_later_than_session_median': int((part.threshold_peak_s > part.session_first_lick_s).sum()),
            'fraction_later': float((part.threshold_peak_s > part.session_first_lick_s).mean()),
        })
    return pd.DataFrame(rows)


def peak_scatter(peaks, fits, output):
    time_peaks = peaks[peaks.neuron_type.eq('time_cell')].rename(
        columns={'threshold_peak_s': 'threshold_peak_s'}
    )
    fitted = fits[fits.group.eq('time_cell')][
        ['session', 'neu_id', 'gaussian_mu']
    ]
    points = time_peaks.merge(fitted, on=['session', 'neu_id'], validate='one_to_one')
    points = points.dropna(subset=['threshold_peak_s', 'gaussian_mu'])
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axis = plt.subplots(figsize=(4.8, 4.5))
    axis.scatter(
        points.threshold_peak_s,
        points.gaussian_mu,
        s=9,
        alpha=.35,
        color=COLOURS['time_cell'],
        linewidths=0,
    )
    limits = [
        min(points.threshold_peak_s.min(), points.gaussian_mu.min()),
        max(points.threshold_peak_s.max(), points.gaussian_mu.max()),
    ]
    axis.plot(limits, limits, color='#999999', linewidth=.8)
    axis.set(
        xlabel='Threshold peak time (s)',
        ylabel='Fitted Gaussian μ (s)',
    )
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(output.with_suffix('.png'), dpi=300)
    fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig)
    return points


#%% alpha diagnostics
def power_exponent(values, uppers):
    values = np.asarray(values, dtype=float)
    uppers = np.asarray(uppers, dtype=float)
    keep = np.isfinite(values) & np.isfinite(uppers) & (values >= .35) & (values <= uppers)
    values, uppers = values[keep], uppers[keep]
    if not len(values) or np.any(uppers <= .35):
        raise ValueError('invalid power-law support')
    if np.allclose(uppers, uppers[0]):
        log_ratio = np.log(float(uppers[0]) / .35)

        def nll(alpha):
            beta = 1. - alpha
            if abs(beta) < 1e-7:
                log_integral = np.log(log_ratio)
            elif beta * log_ratio > 50:
                log_integral = beta * log_ratio + np.log1p(-np.exp(-beta * log_ratio)) - np.log(beta)
            else:
                log_integral = np.log(np.expm1(beta * log_ratio) / beta)
            return alpha * np.log(values / .35).sum() + len(values) * log_integral
    else:
        log_integrals = np.log(uppers / .35)

        def nll(alpha):
            beta = 1. - alpha
            if abs(beta) < 1e-7:
                normalisers = log_integrals
            else:
                normalisers = np.log(np.expm1(beta * log_integrals) / beta)
            return alpha * np.log(values / .35).sum() + normalisers.sum()

    result = minimize_scalar(nll, bracket=(-2., 4.))
    if not result.success:
        raise ValueError('power-law likelihood did not converge')
    return float(result.x), len(values)


def bootstrap_power(values, uppers, seed=20261107):
    values = np.asarray(values, dtype=float)
    uppers = np.asarray(uppers, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(1000):
        indices = rng.integers(0, len(values), len(values))
        estimates.append(power_exponent(values[indices], uppers[indices])[0])
    estimate, n = power_exponent(values, uppers)
    return estimate, *np.percentile(estimates, [2.5, 97.5]), n


def alpha_table(fits, first_lick):
    selected = fits[fits.group.eq('time_cell') & fits.gaussian_mu.notna()].copy()
    selected['session_median_first_lick_s'] = selected.session.map(first_lick)
    current = bootstrap_power(selected.gaussian_mu.to_numpy(), np.full(len(selected), 7.))
    session_support = bootstrap_power(
        selected.gaussian_mu.to_numpy(),
        selected.session_median_first_lick_s.to_numpy(),
    )
    return pd.DataFrame([
        {
            'support': '0.35 to 7 s',
            'cell_set': 'time cells (n = 1177), overlap excluded',
            'n_cells': len(fits[fits.group.eq('time_cell')]),
            'n_supported': current[3],
            'alpha': current[0],
            'bootstrap_low': current[1],
            'bootstrap_high': current[2],
        },
        {
            'support': '0.35 to session median first lick',
            'cell_set': 'time cells (n = 1177), overlap excluded',
            'n_cells': len(fits[fits.group.eq('time_cell')]),
            'n_supported': session_support[3],
            'alpha': session_support[0],
            'bootstrap_low': session_support[1],
            'bootstrap_high': session_support[2],
        },
    ])


#%% width diagnostics
def matlab_mode_smallest(values):
    unique, counts = np.unique(values, return_counts=True)
    return unique[np.flatnonzero(counts == counts.max())[0]]


def matlab_lowpass(values):
    sample_rate = 500.
    passband = 1.
    transition = (.99 - .98 * .85) * (sample_rate / 2. - passband)
    cutoff = (passband + transition / 2.) / (sample_rate / 2.)
    taps = firwin(61, cutoff, window=('kaiser', kaiser_beta(60.)))
    return np.convolve(values, taps, mode='same')


def calculate_widths(profiles, baseline):
    widths = []
    for row in profiles:
        mean_rate = row[1499:]
        filtered = matlab_lowpass(mean_rate)
        rate_segment = filtered[:3500]
        base = matlab_mode_smallest(rate_segment) if baseline == 'mode' else rate_segment.min()
        threshold = base + .5 * rate_segment.std()
        peak = min(int(np.argmax(mean_rate)), 3499)
        left = peak
        while left > 0 and rate_segment[left] > threshold:
            left -= 1
        if left == peak:
            left = 0
        right = peak
        while right < 3499 and rate_segment[right] > threshold:
            right += 1
        if right == peak:
            right = 3499
        widths.append((right - left + 1) * .002)
    return np.asarray(widths)


def width_table(root, raw_root):
    saved = pd.read_csv(root / 'outputs' / 'width_summary' / 'widths.csv')
    tic_rows = saved[
        (saved.subset == 'all') &
        saved.neuron_type.isin(['tic_down', 'tic_up', 'tic_and_time_cell'])
    ].reset_index(drop=True)
    field_rows = saved[
        (saved.subset == 'all') &
        saved.neuron_type.eq('time_cell')
    ].reset_index(drop=True)
    table_root = raw_root / 'immobile_code' / 'RiseDown' / 'tables' / '2025-05to05' / 'Variable' / 'Path4sData'
    tic_profiles = load_profiles(table_root / 'FR_Prof' / 'PAC' / 'FRprofileTable_alignLastLickdffgfthres.mat')
    field_profiles = load_profiles(table_root / 'Fields' / 'FR_Profile' / 'FRprofileTable_alignLastLickdffgf.mat')
    rows = []
    for group_name, frame, profiles in [
        ('TICs', tic_rows, tic_profiles),
        ('Time cells', field_rows, field_profiles),
    ]:
        for baseline in ['mode', 'minimum']:
            reproduced = calculate_widths(profiles, baseline)
            saved_values = frame.width.to_numpy()
            difference = np.abs(reproduced - saved_values)
            rows.append({
                'group': group_name,
                'baseline': baseline,
                'n_cells': len(saved_values),
                'fraction_within_one_bin': float((difference <= .00201).mean()),
                'median_abs_error_s': float(np.median(difference)),
                'mean_abs_error_s': float(np.mean(difference)),
                'max_abs_error_s': float(np.max(difference)),
            })
    tic_saved = tic_rows.width.to_numpy()
    rows.append({
        'group': 'TICs',
        'baseline': 'saved widths > 4 s',
        'n_cells': len(tic_saved),
        'fraction_within_one_bin': float((tic_saved > 4.).mean()),
        'median_abs_error_s': np.nan,
        'mean_abs_error_s': np.nan,
        'max_abs_error_s': np.nan,
    })
    return pd.DataFrame(rows)


#%% trial exclusions
def trial_exclusion_table(root):
    checks = pd.read_csv(root / 'outputs' / 'fits_v1_1' / 'dataset_checks.csv')
    return pd.DataFrame({
        'session': checks.session,
        'all_trials': checks.source_trial_rows,
        'kept_trials': checks.valid_trials,
        'excluded_trials': checks.source_trial_rows - checks.valid_trials,
    })


#%% run
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    root = args.repo_root
    raw_root = args.raw_root
    output = root / 'outputs' / 'diagnostics'
    figures = root / 'figures' / 'wider_evidence'
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    order_file = raw_root / 'immobile_code' / 'Paths' / 'PathFor4sData.m'
    session_order = load_session_order(order_file)
    first_lick = load_first_lick_medians(
        raw_root / 'immobile_code' / 'RiseDown' / 'tables' / '2025-05to05' /
        'Variable' / 'Path4sData' / 'FirstLickMedSessions.mat', session_order
    )
    fits = load_wider_parameters(root)
    widths = pd.read_csv(root / 'outputs' / 'width_summary' / 'widths.csv')
    peaks = peak_table(
        widths,
        load_vector(raw_root / 'PaperNov2026_CurrentBio' / 'ReviewAnalysis' / 'Width' / 'FieldPeakAllTr.mat'),
        load_vector(raw_root / 'PaperNov2026_CurrentBio' / 'ReviewAnalysis' / 'Width' / 'TICPeakAllTr.mat'),
    )
    fractions = peak_fraction_table(peaks, first_lick)
    points = peak_scatter(
        peaks,
        fits,
        figures / 'threshold_peak_vs_gaussian_mu_time_cells',
    )
    alpha = alpha_table(fits, first_lick)
    width = width_table(root, raw_root)
    exclusions = trial_exclusion_table(root)

    fractions.to_csv(output / 'threshold_peak_fractions.csv', index=False)
    points.to_csv(output / 'threshold_peak_vs_gaussian_mu_time_cells.csv', index=False)
    alpha.to_csv(output / 'alpha_support.csv', index=False)
    width.to_csv(output / 'width_reproduction.csv', index=False)
    exclusions.to_csv(output / 'trial_exclusions.csv', index=False)
    report = [
        '# read-only diagnostics',
        '',
        '## session order source',
        str(order_file),
        '',
        '## threshold peak fractions',
        fractions.to_string(index=False),
        '',
        '## alpha support',
        alpha.to_string(index=False),
        '',
        '## width reproduction',
        width.to_string(index=False),
        '',
        '## trial exclusions',
        exclusions.to_string(index=False),
    ]
    (output / 'diagnostics_report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('\n'.join(report))


if __name__ == '__main__':
    main()
