'''
2026-11-07
compare wider-window parameters with published time-cell and temporal-context benchmarks
@author: Dinghao Luo
'''

#%% imports
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WIDER = ROOT / 'outputs' / 'undecided_check'
OUTPUT = ROOT / 'outputs' / 'wider_evidence'
FIGURES = ROOT / 'figures' / 'wider_evidence'


def power_exponent(values):
    from analysis.summarise_fits_base_v1_2 import power_exponent
    return power_exponent(values)


def bootstrap_alpha(values, seed=20261107):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sessions = np.arange(len(values))
    estimates = []
    for _ in range(1000):
        sampled = values[rng.integers(0, len(values), len(values))]
        estimates.append(power_exponent(sampled))
    return power_exponent(values), *np.percentile(estimates, [2.5, 97.5])


def load_parameters():
    return pd.concat([pd.read_csv(path) for path in WIDER.glob('*_wider_parameters.csv')], ignore_index=True)


def box_values(frame, column, group, unbound=True):
    part = frame[frame.group.eq(group)].copy()
    if unbound:
        part = part[~part[f'{column}_at_bound']]
    return part[column].dropna().to_numpy()


def make_figure(parameters, summary):
    from matplotlib.lines import Line2D
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'pdf.use14corefonts': False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colours = {'tic_down': '#4477AA', 'tic_up': '#CC6677', 'time_cell': '#66AA55'}

    # hippocampal time-cell benchmarks
    time_sigma = box_values(parameters, 'gaussian_sigma', 'time_cell', unbound=False)
    boxes = axes[0, 0].boxplot([time_sigma], positions=[1], widths=.45, patch_artist=True, showfliers=False,
                               medianprops={'color': 'black', 'linewidth': 1.5})
    boxes['boxes'][0].set(facecolor='#66AA55', alpha=.6)
    axes[0, 0].hlines(.69, .65, 1.35, color='#AA4499', linewidth=2.5)
    axes[0, 0].set(xticks=[1], xticklabels=['Time cells'], ylabel='Gaussian width (s)',
                   title='Hippocampal time-cell width')
    axes[0, 0].legend(handles=[Line2D([0], [0], color='#AA4499', linewidth=2.5, label='Cao et al. 2022')],
                      frameon=False, fontsize=8)

    alpha, alpha_low, alpha_high = summary['alpha']
    axes[0, 1].errorbar([0], [alpha], yerr=[[alpha - alpha_low], [alpha_high - alpha]],
                        fmt='o', color='#66AA55', capsize=5, label='Time cells')
    axes[0, 1].errorbar([1], [.92], yerr=[[.92 - .73], [1.11 - .92]],
                        fmt='_', color='#AA4499', markersize=18, capsize=5, label='Cao et al. 2022')
    axes[0, 1].set(xticks=[0, 1], xticklabels=['Time cells', 'Cao et al. 2022'],
                   ylabel='peak-time power exponent', title='Time-cell peak distribution')
    axes[0, 1].legend(frameon=False, fontsize=8)

    # temporal-context benchmark
    tau_down = box_values(parameters, 'decay_tau', 'tic_down', unbound=True)
    tau_up = box_values(parameters, 'rise_tau', 'tic_up', unbound=True)
    boxes = axes[1, 0].boxplot([tau_down, tau_up], positions=[1, 2], widths=.45, patch_artist=True,
                               showfliers=False, medianprops={'color': 'black', 'linewidth': 1.5})
    for box, colour in zip(boxes['boxes'], [colours['tic_down'], colours['tic_up']]):
        box.set(facecolor=colour, alpha=.6)
    axes[1, 0].errorbar([1, 2], [2.25, 1.03], yerr=[[2.25 - .59, 1.03 - .26], [4. - 2.25, 3. - 1.03]],
                        fmt='_', color='#222222', markersize=15, capsize=4, label='Cao et al. 2024')
    axes[1, 0].set(xticks=[1, 2], xticklabels=['ramp-down TIC', 'ramp-up TIC'], ylabel='tau (s)',
                   title='Temporal-context tau comparison')
    axes[1, 0].legend(frameon=False, fontsize=8)

    mu_down = box_values(parameters, 'decay_mu', 'tic_down', unbound=True)
    mu_up = box_values(parameters, 'rise_mu', 'tic_up', unbound=True)
    boxes = axes[1, 1].boxplot([mu_down, mu_up], positions=[1, 2], widths=.45, patch_artist=True,
                               showfliers=False, medianprops={'color': 'black', 'linewidth': 1.5})
    for box, colour in zip(boxes['boxes'], [colours['tic_down'], colours['tic_up']]):
        box.set(facecolor=colour, alpha=.6)
    axes[1, 1].errorbar([1, 2], [.01, -.02], yerr=[[.01 - 0., -.02 - (-.13)], [.10 - .01, -.01 - (-.02)]],
                        fmt='_', color='#222222', markersize=15, capsize=4, label='Cao et al. 2024')
    axes[1, 1].set(xticks=[1, 2], xticklabels=['ramp-down TIC', 'ramp-up TIC'], ylabel='event-relative peak (s)',
                   title='Temporal-context peak comparison')
    axes[1, 1].legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    fig.suptitle('wider-window parameters and published benchmarks')
    fig.tight_layout()
    fig.savefig(FIGURES / 'paper_parameter_comparison.png', dpi=300)
    fig.savefig(FIGURES / 'paper_parameter_comparison.pdf')
    plt.close(fig)


def main():
    parameters = load_parameters()
    time_mu = parameters.loc[parameters.group.eq('time_cell') & ~parameters.gaussian_mu_at_bound, 'gaussian_mu'].to_numpy()
    alpha = bootstrap_alpha(time_mu)
    rows = [
        dict(quantity='time-cell Gaussian width', this_study_median=np.median(box_values(parameters, 'gaussian_sigma', 'time_cell', False)),
             published_median=.69, published_low=np.nan, published_high=np.nan, note='parameter definitions differ'),
        dict(quantity='time-cell peak power exponent', this_study_median=alpha[0], this_study_low=alpha[1], this_study_high=alpha[2],
             published_median=.92, published_low=.73, published_high=1.11, note='truncated power-law fit'),
    ]
    for group, shape, label, median, low, high in [('tic_down', 'decay', 'past', 2.25, .59, 4.),
                                                     ('tic_up', 'rise', 'future', 1.03, .26, 3.)]:
        values_tau = box_values(parameters, f'{shape}_tau', group, True)
        values_mu = box_values(parameters, f'{shape}_mu', group, True)
        rows.extend([
            dict(quantity=f'{label} tau', group=group, this_study_median=np.median(values_tau), this_study_low=np.percentile(values_tau, 25),
                 this_study_high=np.percentile(values_tau, 75), published_median=median, published_low=low, published_high=high,
                 bound_pct=100 * parameters.loc[parameters.group.eq(group), f'{shape}_tau_at_bound'].mean(),
                 note='tau-bound-excluded wider-window values'),
            dict(quantity=f'{label} peak', group=group, this_study_median=np.median(values_mu), this_study_low=np.percentile(values_mu, 25),
                 this_study_high=np.percentile(values_mu, 75), published_median=np.nan, published_low=np.nan, published_high=np.nan,
                 bound_pct=100 * parameters.loc[parameters.group.eq(group), f'{shape}_mu_at_bound'].mean(),
                 note='published peak comparison shown in figure'),
        ])
    summary = {'alpha': alpha}
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / 'paper_parameter_comparison.csv', index=False)
    (OUTPUT / 'paper_parameter_comparison.md').write_text(
        '# paper parameter comparison\n\n' +
        'The figure compares quantities with matching event-relative definitions where possible. The wider-window calcium distributions use the wider-window calcium fits. Published values are shown as benchmarks; they were obtained with different data modalities and fitting models. Tau distributions use the tau-bound-excluded wider-window values and report the bound rate separately.\n\n' +
        table.to_string(index=False), encoding='utf-8')
    make_figure(parameters, summary)
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
