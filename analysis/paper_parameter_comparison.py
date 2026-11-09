'''
2026-11-09
compare fitted receptive-field parameters with published summaries
@author: Dinghao Luo
'''

#%% imports
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.optimize import minimize_scalar
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WIDER = ROOT / 'outputs' / 'undecided_check'
OUTPUT = ROOT / 'outputs' / 'wider_evidence'
FIGURES = ROOT / 'figures' / 'wider_evidence'

GROUPS = ['tic_down', 'tic_up', 'time_cell']
LABELS = {
    'tic_down': 'Ramp-down TIC',
    'tic_up': 'Ramp-up TIC',
    'time_cell': 'Time cells',
}
COLOURS = {
    'tic_down': '#228833',
    'tic_up': '#4477AA',
    'time_cell': '#9467BD',
    'published': '#999999',
}
PUBLISHED = {
    'alpha': dict(label='Published time cells', median=.92, q25=.73, q75=1.11),
    'tau_down': dict(label='Published past', median=2.25, q25=.59, q75=4.),
    'tau_up': dict(label='Published future', median=1.03, q25=.26, q75=3.),
    'mu_down': dict(label='Published past', median=.01, q25=0., q75=.10),
    'mu_up': dict(label='Published future', median=-.02, q25=-.13, q75=-.01),
}


#%% data helpers
def power_exponent(values, upper=7.):
    values = np.asarray(values, dtype=float)
    values = values[(values >= .35) & (values <= upper) & np.isfinite(values)]
    log_ratio = np.log(upper / .35)

    def nll(alpha):
        beta = 1 - alpha
        if abs(beta) < 1e-7:
            log_integral = np.log(log_ratio)
        elif beta * log_ratio > 50:
            log_integral = beta * log_ratio + np.log1p(-np.exp(-beta * log_ratio)) - np.log(beta)
        else:
            log_integral = np.log(np.expm1(beta * log_ratio) / beta)
        return alpha * np.log(values / .35).sum() + len(values) * log_integral

    result = minimize_scalar(nll, bracket=(-2., 4.))
    if not result.success:
        raise ValueError('power-law likelihood did not converge')
    return float(result.x)


def bootstrap_alpha(values, upper=7., seed=20261107):
    values = np.asarray(values, dtype=float)
    values = values[(values >= .35) & (values <= upper) & np.isfinite(values)]
    rng = np.random.default_rng(seed)
    estimates = np.array([
        power_exponent(values[rng.integers(0, len(values), len(values))], upper)
        for _ in range(1000)
    ])
    return power_exponent(values, upper), *np.percentile(estimates, [2.5, 97.5])


def load_parameters():
    paths = sorted(WIDER.glob('*_wider_parameters.csv'))
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def fitted_values(parameters, group, quantity):
    part = parameters[parameters.group.eq(group)].copy()
    if quantity == 'mu':
        parameter = 'gaussian_mu'
        bound = 'gaussian_mu_at_bound'
    elif quantity == 'peak':
        parameter = 'rise_mu' if group == 'tic_up' else 'decay_mu'
        bound = f'{parameter}_at_bound'
    elif quantity == 'sigma':
        parameter = 'gaussian_sigma'
        bound = 'gaussian_sigma_at_bound'
    else:
        parameter = 'rise_tau' if group == 'tic_up' else 'decay_tau'
        bound = f'{parameter}_at_bound'
    return part.loc[~part[bound].astype(bool), parameter].dropna().to_numpy(), parameter


def session_reference_test(parameters, group, quantity, reference):
    part = parameters[parameters.group.eq(group)].copy()
    values, parameter = fitted_values(part, group, quantity)
    part = part.loc[~part[f'{parameter}_at_bound'].astype(bool)]
    session_values = part.groupby('session')[parameter].median().dropna()
    result = wilcoxon(session_values.to_numpy() - reference)
    return dict(
        n_sessions=len(session_values),
        session_median=float(session_values.median()),
        p=float(result.pvalue),
    )


def summary_box(axis, y, median, q25, q75, colour):
    height = .36
    axis.add_patch(Rectangle((q25, y - height / 2), q75 - q25, height,
                             facecolor=colour, edgecolor=colour, alpha=.60))
    axis.plot([median, median], [y - height / 2, y + height / 2],
              color='black', linewidth=1.3)


def data_box(axis, y, data, colour):
    axis.boxplot(
        [np.asarray(data, dtype=float)],
        positions=[y],
        vert=False,
        widths=.36,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        boxprops={'facecolor': colour, 'edgecolor': colour, 'alpha': .60},
        whiskerprops={'color': colour, 'linewidth': 1.2},
        capprops={'color': colour, 'linewidth': 1.2},
        medianprops={'color': 'black', 'linewidth': 1.3},
    )


def format_axis(axis):
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.tick_params(length=3, width=.8)


#%% figure
def make_figure(parameters, alpha, alpha_bootstrap):
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'pdf.use14corefonts': False,
        'font.size': 9,
        'axes.titlesize': 11,
        'axes.labelsize': 9,
    })

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), constrained_layout=True)

    axis = axes[0]
    axis.set_title('Peak-time exponent')
    data_box(axis, 1, alpha_bootstrap, COLOURS['time_cell'])
    ref = PUBLISHED['alpha']
    summary_box(axis, 0, ref['median'], ref['q25'], ref['q75'], COLOURS['published'])
    axis.set(
        yticks=[0, 1],
        yticklabels=[ref['label'], 'Time cells'],
        xlabel='Exponent, α',
        xlim=(.55, 1.7),
    )
    format_axis(axis)

    axis = axes[1]
    axis.set_title('Time constant')
    tau_rows = [
        (PUBLISHED['tau_down'], 0, None, COLOURS['published']),
        ({'label': LABELS['tic_down']}, 1, fitted_values(parameters, 'tic_down', 'tau')[0], COLOURS['tic_down']),
        (PUBLISHED['tau_up'], 2, None, COLOURS['published']),
        ({'label': LABELS['tic_up']}, 3, fitted_values(parameters, 'tic_up', 'tau')[0], COLOURS['tic_up']),
    ]
    for row, y, data, colour in tau_rows:
        if data is None:
            summary_box(axis, y, row['median'], row['q25'], row['q75'], colour)
        else:
            data_box(axis, y, data, colour)
    axis.set(
        yticks=[0, 1, 2, 3],
        yticklabels=[row[0]['label'] for row in tau_rows],
        xlabel='τ (s)',
        xlim=(0, 10.5),
    )
    format_axis(axis)

    axis = axes[2]
    axis.set_title('Fitted peak location')
    mu_rows = [
        (PUBLISHED['mu_down'], 0, None, COLOURS['published']),
        ({'label': LABELS['tic_down']}, 1, fitted_values(parameters, 'tic_down', 'peak')[0], COLOURS['tic_down']),
        (PUBLISHED['mu_up'], 2, None, COLOURS['published']),
        ({'label': LABELS['tic_up']}, 3, fitted_values(parameters, 'tic_up', 'peak')[0], COLOURS['tic_up']),
    ]
    for row, y, data, colour in mu_rows:
        if data is None:
            summary_box(axis, y, row['median'], row['q25'], row['q75'], colour)
        else:
            data_box(axis, y, data, colour)
    axis.axvline(0, color='#999999', linewidth=.8)
    axis.set(
        yticks=[0, 1, 2, 3],
        yticklabels=[row[0]['label'] for row in mu_rows],
        xlabel='μ (s)',
        xlim=(-1.05, 1.05),
    )
    format_axis(axis)

    for label, axis in zip('ABC', axes):
        axis.text(-.12, 1.05, label, transform=axis.transAxes,
                  fontsize=12, fontweight='bold', va='top')

    fig.savefig(FIGURES / 'published_parameter_comparison.png', dpi=300)
    fig.savefig(FIGURES / 'published_parameter_comparison.pdf')
    plt.close(fig)


#%% summary
def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    parameters = load_parameters()

    time_peaks = parameters.loc[parameters.group.eq('time_cell'), 'gaussian_mu'].dropna().to_numpy()
    alpha = bootstrap_alpha(time_peaks)

    reference_tests = {
        ('tic_down', 'tau'): session_reference_test(parameters, 'tic_down', 'tau', PUBLISHED['tau_down']['median']),
        ('tic_up', 'tau'): session_reference_test(parameters, 'tic_up', 'tau', PUBLISHED['tau_up']['median']),
        ('tic_down', 'peak'): session_reference_test(parameters, 'tic_down', 'peak', PUBLISHED['mu_down']['median']),
        ('tic_up', 'peak'): session_reference_test(parameters, 'tic_up', 'peak', PUBLISHED['mu_up']['median']),
    }

    rows = [{
        'quantity': 'peak-time exponent',
        'group': 'Time cells',
        'source': 'gaussian_mu',
        'n_cells': int(parameters.loc[parameters.group.eq('time_cell'), 'session'].count()),
        'n_supported': int(((time_peaks >= .35) & (time_peaks <= 7.)).sum()),
        'cell_set': 'time cells (n = 1177), overlap excluded',
        'n_sessions': int(parameters.loc[parameters.group.eq('time_cell'), 'session'].nunique()),
        'median': alpha[0],
        'q25': alpha[1],
        'q75': alpha[2],
        'published_median': PUBLISHED['alpha']['median'],
        'published_q25': PUBLISHED['alpha']['q25'],
        'published_q75': PUBLISHED['alpha']['q75'],
        'one_sample_p': np.nan,
    }]
    for group, quantity, key in [
        ('tic_down', 'tau', 'tau_down'),
        ('tic_up', 'tau', 'tau_up'),
        ('tic_down', 'peak', 'mu_down'),
        ('tic_up', 'peak', 'mu_up'),
    ]:
        values, parameter = fitted_values(parameters, group, quantity)
        reference = PUBLISHED[key]
        test = reference_tests[(group, quantity)]
        rows.append({
            'quantity': 'time constant' if quantity == 'tau' else 'fitted peak location',
            'group': LABELS[group],
            'source': parameter,
            'n_cells': len(parameters.loc[parameters.group.eq(group)]),
            'n_supported': len(values),
            'cell_set': 'fitted cells with bound values excluded',
            'n_sessions': test['n_sessions'],
            'median': float(np.median(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75)),
            'published_median': reference['median'],
            'published_q25': reference['q25'],
            'published_q75': reference['q75'],
            'one_sample_p': test['p'],
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / 'paper_parameter_comparison.csv', index=False)

    report = [
        '# fitted parameter comparison',
        '',
        'The data summaries use the fitted Gaussian and directional-exponential parameters. Published rows contain the reported median and interquartile range.',
        '',
        table.to_string(index=False),
    ]
    (OUTPUT / 'paper_parameter_comparison.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    make_figure(parameters, alpha, bootstrap_alpha(time_peaks)[1:])


if __name__ == '__main__':
    main()
