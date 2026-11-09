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
from utils.rf_models_v1_2 import BOUNDS

WIDER = ROOT / 'outputs' / 'undecided_check'
OUTPUT = ROOT / 'outputs' / 'wider_evidence'
FIGURES = ROOT / 'figures' / 'wider_evidence'

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
    'alpha': dict(label='Cao et al. 2022', median=.92, q25=.73, q75=1.11,
                  interval='95% credible interval'),
    'tau_down': dict(label='Cao et al. 2024 past', median=2.25, q25=.59, q75=4.),
    'tau_up': dict(label='Cao et al. 2024 future', median=1.03, q25=.26, q75=3.),
    'mu_down': dict(label='Cao et al. 2024 past', median=.01, q25=0., q75=.10),
    'mu_up': dict(label='Cao et al. 2024 future', median=-.02, q25=-.13, q75=-.01),
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


def parameter_name(group, quantity):
    if quantity == 'mu':
        return 'gaussian_mu'
    if quantity == 'peak':
        return 'rise_mu' if group == 'tic_up' else 'decay_mu'
    return 'rise_tau' if group == 'tic_up' else 'decay_tau'


def display_parameter_name(group, quantity):
    parameter = parameter_name(group, quantity)
    return '-rise_mu' if group == 'tic_up' and quantity == 'peak' else parameter


def raw_parameter_name(parameter):
    return parameter[1:] if parameter.startswith('-') else parameter


def fitted_values(parameters, group, quantity, include_bound=False):
    part = parameters[parameters.group.eq(group)].copy()
    parameter = parameter_name(group, quantity)
    bound = f'{parameter}_at_bound'
    if not include_bound:
        part = part.loc[~part[bound].astype(bool)]
    values = part.loc[part[parameter].notna(), parameter].to_numpy()
    if group == 'tic_up' and quantity == 'peak':
        values = -values
    return values, display_parameter_name(group, quantity)


def bound_percentages(parameters, group, parameter):
    parameter = raw_parameter_name(parameter)
    part = parameters[parameters.group.eq(group)]
    values = part.loc[part[f'{parameter}_at_bound'].astype(bool), parameter].dropna().to_numpy()
    if not len(values):
        return 0., 0.
    model = 'gaussian' if parameter.startswith('gaussian_') else 'rise' if parameter.startswith('rise_') else 'decay'
    limit_index = 0 if parameter.endswith('_mu') else 1
    lower_limit, upper_limit = BOUNDS[model][0][limit_index], BOUNDS[model][1][limit_index]
    lower = np.isclose(values, lower_limit, rtol=0., atol=1e-5)
    upper = np.isclose(values, upper_limit, rtol=0., atol=1e-5)
    denominator = len(part)
    return 100. * lower.sum() / denominator, 100. * upper.sum() / denominator


def session_reference_test(parameters, group, quantity, reference, include_bound):
    part = parameters[parameters.group.eq(group)].copy()
    parameter = parameter_name(group, quantity)
    bound = f'{parameter}_at_bound'
    if not include_bound:
        part = part.loc[~part[bound].astype(bool)]
    values = part[parameter].copy()
    if group == 'tic_up' and quantity == 'peak':
        values = -values
    session_values = values.groupby(part['session']).median().dropna()
    return dict(
        n_sessions=len(session_values),
        session_median=float(session_values.median()),
        p=float(wilcoxon(session_values.to_numpy() - reference).pvalue),
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
def make_figure(parameters):
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'pdf.use14corefonts': False,
        'font.size': 9,
        'axes.titlesize': 11,
        'axes.labelsize': 9,
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8), constrained_layout=True)

    axis = axes[0]
    axis.set_title('Time constant')
    tau_rows = [
        (PUBLISHED['tau_down'], 0, None, COLOURS['published']),
        ({'label': LABELS['tic_down']}, 1, fitted_values(parameters, 'tic_down', 'tau', True)[0], COLOURS['tic_down']),
        (PUBLISHED['tau_up'], 2, None, COLOURS['published']),
        ({'label': LABELS['tic_up']}, 3, fitted_values(parameters, 'tic_up', 'tau', True)[0], COLOURS['tic_up']),
    ]
    for row, y, data, colour in tau_rows:
        if data is None:
            summary_box(axis, y, row['median'], row['q25'], row['q75'], colour)
        else:
            data_box(axis, y, data, colour)
    axis.set(yticks=[0, 1, 2, 3],
             yticklabels=[row[0]['label'] for row in tau_rows],
             xlabel='τ (s)', xlim=(0, 10.5))
    format_axis(axis)

    axis = axes[1]
    axis.set_title('Fitted peak location')
    peak_rows = [
        (PUBLISHED['mu_down'], 0, None, COLOURS['published']),
        ({'label': LABELS['tic_down']}, 1, fitted_values(parameters, 'tic_down', 'peak', True)[0], COLOURS['tic_down']),
        (PUBLISHED['mu_up'], 2, None, COLOURS['published']),
        ({'label': LABELS['tic_up']}, 3, fitted_values(parameters, 'tic_up', 'peak', True)[0], COLOURS['tic_up']),
    ]
    for row, y, data, colour in peak_rows:
        if data is None:
            summary_box(axis, y, row['median'], row['q25'], row['q75'], colour)
        else:
            data_box(axis, y, data, colour)
    axis.axvline(0, color='#999999', linewidth=.8)
    axis.set(yticks=[0, 1, 2, 3],
             yticklabels=[row[0]['label'] for row in peak_rows],
             xlabel='μ (s)', xlim=(-1.05, 1.05))
    format_axis(axis)

    for label, axis in zip('AB', axes):
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

    rows = [{
        'quantity': 'peak-time exponent',
        'group': 'Time cells',
        'source': 'gaussian_mu',
        'n_cells_including_bounds': len(parameters.loc[parameters.group.eq('time_cell')]),
        'n_cells_in_support': int(((time_peaks >= .35) & (time_peaks <= 7.)).sum()),
        'cell_set': 'time cells (n = 1177), overlap excluded',
        'alpha': alpha[0],
        'alpha_low': alpha[1],
        'alpha_high': alpha[2],
        'alpha_interval': PUBLISHED['alpha']['interval'],
        'published_median': PUBLISHED['alpha']['median'],
        'published_q25': PUBLISHED['alpha']['q25'],
        'published_q75': PUBLISHED['alpha']['q75'],
    }]
    for group, quantity, key in [
        ('tic_down', 'tau', 'tau_down'),
        ('tic_up', 'tau', 'tau_up'),
        ('tic_down', 'peak', 'mu_down'),
        ('tic_up', 'peak', 'mu_up'),
    ]:
        values_in, parameter = fitted_values(parameters, group, quantity, include_bound=True)
        values_ex, _ = fitted_values(parameters, group, quantity, include_bound=False)
        reference = PUBLISHED[key]
        lower_pct, upper_pct = bound_percentages(parameters, group, parameter)
        tests = {
            'including': session_reference_test(parameters, group, quantity, reference['median'], True),
            'excluding': session_reference_test(parameters, group, quantity, reference['median'], False),
        }
        row = {
            'quantity': 'time constant' if quantity == 'tau' else 'fitted peak location',
            'group': LABELS[group],
            'source': parameter,
            'convention': ('-rise_mu, relative to interval end' if group == 'tic_up' and quantity == 'peak'
                           else 'decay_mu, relative to last lick' if group == 'tic_down' and quantity == 'peak'
                           else ''),
            'n_cells_including_bounds': len(values_in),
            'n_cells_excluding_bounds': len(values_ex),
            'median_including_bounds': float(np.median(values_in)),
            'median_excluding_bounds': float(np.median(values_ex)),
            'p_including_bounds': tests['including']['p'],
            'p_excluding_bounds': tests['excluding']['p'],
            'pct_lower_bound': lower_pct,
            'pct_upper_bound': upper_pct,
            'published_median': reference['median'],
            'published_q25': reference['q25'],
            'published_q75': reference['q75'],
        }
        if quantity == 'tau':
            row['pct_tau_gt9_s'] = 100. * float((values_in > 9.).mean())
            row['cao_past_excluded_gt9'] = '16/331' if group == 'tic_down' else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / 'paper_parameter_comparison.csv', index=False)

    report = [
        '# fitted parameter comparison',
        '',
        f"Published alpha limits are the {PUBLISHED['alpha']['interval']}.",
        'The parameter tables report censored values at their fitted bounds and the corresponding bound-excluded summaries.',
        '',
        table.to_string(index=False),
    ]
    (OUTPUT / 'paper_parameter_comparison.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    make_figure(parameters)


if __name__ == '__main__':
    main()
