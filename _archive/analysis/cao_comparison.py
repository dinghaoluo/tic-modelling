'''
2026-11-06
make a bound-aware comparison with Cao's past and future receptive fields
@author: Dinghao Luo
'''

#%% imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FITS = ROOT / 'outputs' / 'fits_v1_2' / 'fits_all.csv'
WIDE = ROOT / 'outputs' / 'model_comparison' / 'summary_table_wider.csv'
LITERATURE = ROOT / 'outputs' / 'model_comparison' / 'literature.csv'
OUTPUT = ROOT / 'outputs' / 'model_comparison'
FIGURES = ROOT / 'figures' / 'model_comparison'

CAO = {
    'tic_down': {'label': 'past', 'mu': .01, 'mu_q25': 0., 'mu_q75': .10,
                 'tau': 2.25, 'tau_q25': .59, 'tau_q75': 4.00},
    'tic_up': {'label': 'future', 'mu': -.02, 'mu_q25': -.13, 'mu_q75': -.01,
               'tau': 1.03, 'tau_q25': .26, 'tau_q75': 3.00},
}


def markdown(frame):
    frame = frame.copy()
    columns = list(frame.columns)
    rows = ['| ' + ' | '.join(columns) + ' |',
            '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    rows.extend('| ' + ' | '.join(str(value) for value in row) + ' |'
                for row in frame.itertuples(index=False, name=None))
    return '\n'.join(rows)


def shape_bound(frame, shape, parameter):
    return frame[f'{shape}_{parameter}_at_bound'].astype(bool)


def distribution(values):
    values = pd.Series(values).dropna()
    if not len(values):
        return dict(n=0, median=np.nan, q25=np.nan, q75=np.nan)
    return dict(n=len(values), median=values.median(), q25=values.quantile(.25), q75=values.quantile(.75))


def add_tau_rows(fits):
    rows = []
    for group, shape in [('tic_down', 'decay'), ('tic_up', 'rise')]:
        part = fits[fits.group == group].copy()
        tau = part[f'{shape}_tau']
        tau_bound = shape_bound(part, shape, 'tau')
        for subset, keep in [('all', np.ones(len(part), dtype=bool)),
                             ('tau_bound_excluded', ~tau_bound),
                             ('beats_flat_tau_bound_excluded', part.beats_flat & ~tau_bound)]:
            stats = distribution(tau[keep])
            rows.append(dict(quantity='tau', group=group, parameter=f'{shape}_tau', subset=subset,
                             bound_pct=100 * tau_bound.mean(), cao_reference=CAO[group]['label'],
                             cao_median=CAO[group]['tau'], cao_q25=CAO[group]['tau_q25'],
                             cao_q75=CAO[group]['tau_q75'], **stats))
    return rows


def add_peak_rows(fits):
    rows = []
    for group, shape in [('tic_down', 'decay'), ('tic_up', 'rise')]:
        stats = distribution(fits.loc[fits.group == group, f'{shape}_mu'])
        rows.append(dict(quantity='peak_mu', group=group, parameter=f'{shape}_mu', subset='all',
                         bound_pct=100 * shape_bound(fits[fits.group == group], shape, 'mu').mean(),
                         cao_reference=CAO[group]['label'], cao_median=CAO[group]['mu'],
                         cao_q25=CAO[group]['mu_q25'], cao_q75=CAO[group]['mu_q75'], **stats))
    return rows


def add_rho_rows(fits):
    rows = []
    for group in ['tic_down', 'tic_up', 'time_cell']:
        part = fits[fits.group == group]
        available = part[part.rho.notna()]
        if not len(available):
            continue
        bound = available.rho_at_bound.astype(bool)
        for subset, kept in [('available', available), ('rho_bound_excluded', available[~bound])]:
            stats = distribution(kept.rho)
            rows.append(dict(quantity='rho', group=group, parameter='rho', subset=subset,
                             bound_pct=100 * bound.mean(), cao_reference='same linear interval scaling',
                             cao_median=np.nan, cao_q25=np.nan, cao_q75=np.nan, **stats))
    return rows


def add_time_cell_rows(literature):
    rows = []
    for metric, label in [('peak_power_law_alpha', 'time-cell peak power exponent'),
                          ('sigma_median', 'time-cell Gaussian width'),
                          ('width_on_peak_slope', 'time-cell width-on-peak slope')]:
        row = literature[(literature.metric == metric) & (literature.group == 'time_cell')]
        if len(row):
            value = row.iloc[0]
            rows.append(dict(quantity=label, group='time_cell', parameter=value.parameter,
                             subset='saved literature comparison', bound_pct=np.nan,
                             cao_reference=value.citation, cao_median=value.published_value,
                             cao_q25=value.published_lower if 'published_lower' in value else np.nan,
                             cao_q75=value.published_upper if 'published_upper' in value else np.nan,
                             n=value.n_cells, median=value.our_value, q25=np.nan, q75=np.nan))
    return rows


def add_model_rows(wide):
    selected = wide[wide.measure.eq('Gaussian vs exponential')]
    rows = []
    for _, value in selected.iterrows():
        rows.append(dict(group=value['group'], n=int(value['n cells']),
                         gaussian_better=value['% Gaussian better'],
                         exponential_better=value['% Exponential better'],
                         undecided=value['% No clear difference'],
                         median_contrast=value['median'], p=value['p for median']))
    return rows


def make_figure(tau_rows):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    colours = {'all': '#4477AA', 'tau_bound_excluded': '#228833', 'beats_flat_tau_bound_excluded': '#CC6677'}
    for axis, group in zip(axes, ['tic_down', 'tic_up']):
        part = [row for row in tau_rows if row['group'] == group]
        for index, row in enumerate(part):
            yerr = [[row['median'] - row['q25']], [row['q75'] - row['median']]]
            axis.errorbar(index, row['median'], yerr=yerr, fmt='o', color=colours[row['subset']], capsize=4,
                          label=row['subset'].replace('_', ' '))
        reference = CAO[group]
        axis.errorbar(len(part), reference['tau'],
                      yerr=[[reference['tau'] - reference['tau_q25']], [reference['tau_q75'] - reference['tau']]],
                      fmt='_', markersize=18, color='#CC6677', capsize=4, label='Cao median and IQR')
        bound = next(row['bound_pct'] for row in part if row['subset'] == 'all')
        axis.set_title(f'{group.replace("tic_", "")} (tau bound: {bound:.1f}%)')
        axis.set_xticks(range(len(part) + 1), [row['subset'].replace('_', '\n') for row in part] + ['Cao'])
        axis.tick_params(axis='x', labelsize=8)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    axes[0].set_ylabel('time constant tau (s)')
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles[:4], labels[:4], frameon=False, fontsize=8)
    fig.suptitle('bound-aware comparison with Cao et al. 2024')
    fig.tight_layout()
    fig.savefig(FIGURES / 'cao_method_matched.png', dpi=300)
    plt.close(fig)


def main():
    fits = pd.read_csv(FITS)
    literature = pd.read_csv(LITERATURE)
    wide = pd.read_csv(WIDE)
    tau_rows = add_tau_rows(fits)
    rows = tau_rows + add_peak_rows(fits) + add_rho_rows(fits) + add_time_cell_rows(literature)
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / 'cao_method_matched.csv', index=False)
    model_rows = add_model_rows(wide)
    lines = [
        '# Cao comparison',
        '',
        'The comparison uses the same event-relative quantities where they are defined in both analyses: peak location, exponential time constant tau, and the linear interval-scaling parameter rho. The wider-window calcium tau values come from the calcium-kernel fits and Cao et al. use hierarchical Bayesian spike-rate models, so the numerical comparison is a benchmark rather than a claim of identical estimators.',
        '',
        'Tau rows distinguish all finite fits, fits whose tau is not at the search bound, and the latter subset after requiring the shape to beat the flat guard. The all-fit bound percentage is reported in every row. Because the ramp-down supported subset contains only a small number of cells, I do not fit a power law or mixture to the calcium tau distribution.',
        '',
        '## parameter comparison',
        markdown(table),
        '',
        '## held-out shape comparison',
        'The cell-level held-out comparison is retained as a separate result because Cao et al. used WAIC and selected temporal cells before the model comparison; the percentages below therefore should not be compared as if they were the same statistic.',
        '',
        markdown(pd.DataFrame(model_rows)),
        '',
        '## interpretation',
        'The boundary-localised peaks are directionally comparable to Cao. The tau medians are sensitive to the upper search bound, especially for ramp-up, and the bound-excluded values should be the primary descriptive comparison. The interval-scaling parameter uses the same linear form as Cao, but the calcium rho estimates are sparse and often bound-limited, so the early/late and delay-length manipulations remain the stronger scaling evidence.',
    ]
    (OUTPUT / 'cao_method_matched.md').write_text('\n'.join(lines), encoding='utf-8')
    make_figure(tau_rows)
    print(table.to_string(index=False))
    print(pd.DataFrame(model_rows).to_string(index=False))


if __name__ == '__main__':
    main()
