''' 
2026-11-08
compare receptive-field parameters with published benchmarks
@author: Dinghao Luo
'''

#%% imports
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WIDER = ROOT / 'outputs' / 'undecided_check'
OUTPUT = ROOT / 'outputs' / 'wider_evidence'
FIGURES = ROOT / 'figures' / 'wider_evidence'

COLOURS = {'study': '#4477AA', 'published': '#555555', 'tic': '#4477AA', 'time': '#66AA55'}


#%% data helpers
def power_exponent(values):
    values = np.asarray(values, dtype=float)
    values = values[(values >= .35) & (values <= 7.) & np.isfinite(values)]
    log_ratio = np.log(7. / .35)

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


def bootstrap_alpha(values, seed=20261107):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = [power_exponent(values[rng.integers(0, len(values), len(values))]) for _ in range(1000)]
    return power_exponent(values), *np.percentile(estimates, [2.5, 97.5])


def load_parameters():
    paths = sorted(WIDER.glob('*_wider_parameters.csv'))
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def summary_box(axis, y, median, low, high, colour):
    stats = [{'med': median, 'q1': low, 'q3': high, 'whislo': low, 'whishi': high, 'fliers': []}]
    artists = axis.bxp(
        stats,
        positions=[y],
        vert=False,
        widths=.36,
        patch_artist=True,
        showfliers=False,
        boxprops={'facecolor': colour, 'edgecolor': colour, 'alpha': .65},
        whiskerprops={'color': colour, 'linewidth': 1.2},
        capprops={'color': colour, 'linewidth': 1.2},
        medianprops={'color': 'black', 'linewidth': 1.3},
    )
    return artists


def format_axis(axis):
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.tick_params(length=3, width=.8)


#%% figure
def make_figure(parameters, session_pairs, alpha):
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'pdf.use14corefonts': False,
        'font.size': 9,
        'axes.titlesize': 11,
        'axes.labelsize': 9,
    })

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.2), constrained_layout=True)
    axis_a, axis_b, axis_c, axis_d = axes.flat

    # α is the power-law exponent of the time-cell peak-location distribution
    axis_a.set_title('Time-cell peak locations')
    summary_box(axis_a, 1, alpha[0], alpha[1], alpha[2], COLOURS['study'])
    summary_box(axis_a, 0, .92, .73, 1.11, COLOURS['published'])
    axis_a.set(yticks=[0, 1], yticklabels=['Cao et al. 2022', 'This study'], xlabel='Peak-location exponent, alpha', xlim=(.55, 1.7))
    format_axis(axis_a)

    axis_b.set_title('Time constant (tau)')
    rows_b = [
        ('Cao et al. 2024, past', 0, 2.25, .59, 4., COLOURS['published'], 's'),
        ('Ramp-down TIC', 1, 2.9556, 1.7026, 5.0496, COLOURS['study'], 'o'),
        ('Cao et al. 2024, future', 2, 1.03, .26, 3., COLOURS['published'], 's'),
        ('Ramp-up TIC', 3, 2.5754, 1.2492, 4.4938, COLOURS['study'], 'o'),
    ]
    for label, y, median, low, high, colour, marker in rows_b:
        summary_box(axis_b, y, median, low, high, colour)
    axis_b.set(yticks=[0, 1, 2, 3], yticklabels=[row[0] for row in rows_b], xlabel='τ (s)', xlim=(0, 10.5))
    format_axis(axis_b)

    axis_c.set_title('Peak location relative to interval boundary')
    rows_c = [
        ('Cao et al. 2024, past', 0, .01, 0., .10, COLOURS['published'], 's'),
        ('Ramp-down TIC', 1, .3334, .1282, .7014, COLOURS['study'], 'o'),
        ('Cao et al. 2024, future', 2, -.02, -.13, -.01, COLOURS['published'], 's'),
        ('Ramp-up TIC', 3, .0126, -.0657, .1299, COLOURS['study'], 'o'),
    ]
    for label, y, median, low, high, colour, marker in rows_c:
        summary_box(axis_c, y, median, low, high, colour)
    axis_c.axvline(0, color='#999999', linewidth=.8)
    axis_c.set(yticks=[0, 1, 2, 3], yticklabels=[row[0] for row in rows_c], xlabel='Peak location (s)', xlim=(-.2, .85))
    format_axis(axis_c)

    width = session_pairs[session_pairs.quantity.eq('field_width')]
    axis_d.set_title('Field width within this study')
    for _, row in width.iterrows():
        axis_d.plot([0, 1], [row.time_median, row.tic_median], color='#BBBBBB', linewidth=.8, zorder=1)
    axis_d.scatter(np.zeros(len(width)), width.time_median, color=COLOURS['time'], s=24, zorder=2, label='Time cells')
    axis_d.scatter(np.ones(len(width)), width.tic_median, color=COLOURS['tic'], s=24, zorder=2, label='TICs')
    axis_d.set(xticks=[0, 1], xticklabels=['Time cells', 'TICs'], ylabel='0.5 SD threshold width (s)', xlim=(-.35, 1.35))
    axis_d.text(.03, .93, 'paired sessions, p = 6.1 × 10⁻⁵', transform=axis_d.transAxes, va='top', fontsize=8)
    axis_d.legend(frameon=False, loc='lower right', fontsize=8)
    format_axis(axis_d)

    for label, axis in zip('ABCD', axes.flat):
        axis.text(-.12, 1.05, label, transform=axis.transAxes, fontsize=12, fontweight='bold', va='top')

    fig.savefig(FIGURES / 'paper_parameter_comparison.png', dpi=300)
    fig.savefig(FIGURES / 'paper_parameter_comparison.pdf')
    plt.close(fig)


#%% summary
def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    parameters = load_parameters()
    session_pairs = pd.read_csv(OUTPUT / 'wider_parameter_session_pairs.csv')
    time_peaks = parameters.loc[
        parameters.group.eq('time_cell') & ~parameters.gaussian_mu_at_bound,
        'gaussian_mu',
    ].to_numpy()
    alpha = bootstrap_alpha(time_peaks)

    width = session_pairs[session_pairs.quantity.eq('field_width')]
    width_p = wilcoxon(width.tic_median, width.time_median).pvalue
    tau_p = wilcoxon(
        session_pairs.loc[session_pairs.quantity.eq('tau_unbound'), 'tic_median'],
        session_pairs.loc[session_pairs.quantity.eq('tau_unbound'), 'time_median'],
    ).pvalue
    peak_p = wilcoxon(
        session_pairs.loc[session_pairs.quantity.eq('peak'), 'tic_median'],
        session_pairs.loc[session_pairs.quantity.eq('peak'), 'time_median'],
    ).pvalue

    table = pd.DataFrame([
        {'quantity': 'time-cell peak-location exponent (alpha)', 'this_study': alpha[0], 'this_study_low': alpha[1], 'this_study_high': alpha[2], 'published': .92, 'published_low': .73, 'published_high': 1.11, 'comparison': 'descriptive; methods differ'},
        {'quantity': 'ramp-down time constant (s)', 'this_study': 2.9556, 'this_study_low': 1.7026, 'this_study_high': 5.0496, 'published': 2.25, 'published_low': .59, 'published_high': 4., 'comparison': 'descriptive; calcium vs spikes'},
        {'quantity': 'ramp-up time constant (s)', 'this_study': 2.5754, 'this_study_low': 1.2492, 'this_study_high': 4.4938, 'published': 1.03, 'published_low': .26, 'published_high': 3., 'comparison': 'descriptive; calcium vs spikes'},
        {'quantity': 'ramp-down peak location (s)', 'this_study': .3334, 'this_study_low': .1282, 'this_study_high': .7014, 'published': .01, 'published_low': 0., 'published_high': .10, 'comparison': 'descriptive; boundary definitions aligned'},
        {'quantity': 'ramp-up peak location (s)', 'this_study': .0126, 'this_study_low': -.0657, 'this_study_high': .1299, 'published': -.02, 'published_low': -.13, 'published_high': -.01, 'comparison': 'descriptive; boundary definitions aligned'},
    ])
    table.to_csv(OUTPUT / 'paper_parameter_comparison.csv', index=False)

    report = [
        '# parameter comparison with published benchmarks',
        '',
        'Panels A-C place the wider-window estimates beside the published summary values. The interval is the interquartile range for the cell-level estimates, or the 95% bootstrap interval for alpha in this study; the Cao et al. 2022 interval for alpha is a 95% credible interval.',
        '',
        'The peak-location exponent alpha describes how time-cell peaks are distributed across the interval. In the Cao et al. 2022 model, alpha = 1 corresponds to logarithmic compression. It is not a fit to the shape of an individual cell. The Cao et al. 2024 values are published medians and interquartile ranges, so these cross-paper panels are descriptive and do not carry a new p value.',
        '',
        'Panel D shows the paired-session field-width comparison using the 0.5 SD threshold measure. The paired-session Wilcoxon tests were:',
        '',
        f'- field width: p = {width_p:.3g}',
        f'- matched tau, bound-excluded: p = {tau_p:.3g}',
        f'- Gaussian peak time: p = {peak_p:.3g}',
        '',
        table.to_string(index=False),
    ]
    (OUTPUT / 'paper_parameter_comparison.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    make_figure(parameters, session_pairs, alpha)


if __name__ == '__main__':
    main()
