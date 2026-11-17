'''
2026-11-06
compare pooled TIC and time-cell parameters with session-aware tests
@author: Dinghao Luo
'''

#%% imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
FITS = ROOT / 'outputs' / 'fits_v1_2' / 'fits_all.csv'
WIDTHS = ROOT / 'outputs' / 'width_summary' / 'widths.csv'
CELL_LL = ROOT / 'outputs' / 'model_comparison' / 'cell_model_ll_trial.csv'
OUTPUT = ROOT / 'outputs' / 'model_comparison'
FIGURES = ROOT / 'figures' / 'model_comparison'


def paired_parameter(name, values):
    rows = []
    for session, part in values.groupby('session'):
        tic = part.loc[part.group.eq('tic'), name].dropna()
        time = part.loc[part.group.eq('time'), name].dropna()
        if len(tic) and len(time):
            rows.append(dict(session=session, tic_median=tic.median(), time_median=time.median(),
                             difference=tic.median() - time.median(), n_tic=len(tic), n_time=len(time)))
    paired = pd.DataFrame(rows)
    result = wilcoxon(paired.difference) if len(paired) else None
    return dict(quantity=name, n_sessions=len(paired), pooled_tic=values.loc[values.group.eq('tic'), name].median(),
                pooled_time=values.loc[values.group.eq('time'), name].median(),
                session_difference_median=paired.difference.median(),
                sessions_tic_larger=int((paired.difference > 0).sum()),
                p_wilcoxon=result.pvalue if result is not None else np.nan,
                paired=paired)


def parameter_frames(fits, widths):
    fit = fits.copy()
    original_group = fit.group.copy()
    fit['group'] = np.where(original_group.isin(['tic_down', 'tic_up']), 'tic',
                            np.where(original_group.eq('time_cell'), 'time', 'other'))
    fit['peak'] = fit.gaussian_mu
    fit['tau'] = np.where(original_group.eq('tic_up'), fit.rise_tau, fit.decay_tau)
    tau_bound = np.where(original_group.eq('tic_up'), fit.rise_tau_at_bound, fit.decay_tau_at_bound).astype(bool)
    fit['tau_unbound'] = fit['tau'].where(~tau_bound)
    width = widths[widths.subset.eq('all')].copy()
    width['group'] = np.where(width.neuron_type.isin(['tic_down', 'tic_up']), 'tic',
                              np.where(width.neuron_type.eq('time_cell'), 'time', 'other'))
    width = width.rename(columns={'width': 'field_width'})
    return {
        'field_width': width[['session', 'group', 'field_width']],
        'peak': fit[['session', 'group', 'peak']],
        'tau': fit[['session', 'group', 'tau']],
        'tau_unbound': fit[['session', 'group', 'tau_unbound']],
    }


def model_contrast(cell_ll):
    x = cell_ll.copy()
    x['delta'] = np.nan
    down = x.group.eq('tic_down')
    up = x.group.eq('tic_up')
    time = x.group.eq('time_cell')
    x.loc[down, 'delta'] = x.loc[down, 'gaussian_trial_mean_ll'] - x.loc[down, 'decay_trial_mean_ll']
    x.loc[up, 'delta'] = x.loc[up, 'gaussian_trial_mean_ll'] - x.loc[up, 'rise_trial_mean_ll']
    x.loc[time, 'delta'] = x.loc[time, 'gaussian_trial_mean_ll'] - x.loc[time, ['decay_trial_mean_ll', 'rise_trial_mean_ll']].max(axis=1)
    x['group'] = np.where(x.group.isin(['tic_down', 'tic_up']), 'tic',
                          np.where(x.group.eq('time_cell'), 'time', 'other'))
    paired = []
    for session, part in x.groupby('session'):
        tic = part.loc[part.group.eq('tic'), 'delta'].dropna()
        time_values = part.loc[part.group.eq('time'), 'delta'].dropna()
        if len(tic) and len(time_values):
            paired.append(dict(session=session, tic_median=tic.median(), time_median=time_values.median(),
                               difference=tic.median() - time_values.median()))
    paired = pd.DataFrame(paired)
    statistic = wilcoxon(paired.difference) if len(paired) else None
    return paired, dict(quantity='held-out Gaussian minus expected exponential', n_sessions=len(paired),
                        pooled_tic=x.loc[x.group.eq('tic'), 'delta'].median(),
                        pooled_time=x.loc[x.group.eq('time'), 'delta'].median(),
                        session_difference_median=paired.difference.median(),
                        sessions_tic_larger=int((paired.difference > 0).sum()),
                        p_wilcoxon=statistic.pvalue if statistic is not None else np.nan)


def make_figure(values, results):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    colours = {'tic': '#4477AA', 'time': '#AA55AA'}
    specs = [('field_width', '0.5 SD width (s)'), ('peak', 'Gaussian peak (s)'), ('tau_unbound', 'matched tau (bound excluded, s)')]
    for axis, (name, label) in zip(axes, specs):
        for position, group in enumerate(['tic', 'time']):
            data = values[name].loc[values[name].group.eq(group), name].dropna()
            axis.hist(data, bins=20, density=True, alpha=.6, color=colours[group],
                      label='TICs' if group == 'tic' else 'time cells')
        result = results[name]
        axis.set_xlabel(label)
        axis.set_ylabel('density')
        axis.set_title(f'p = {result["p_wilcoxon"]:.3g}')
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('pooled parameter distributions')
    fig.tight_layout()
    fig.savefig(FIGURES / 'pooled_parameter_distributions.png', dpi=300)
    plt.close(fig)


def main():
    fits = pd.read_csv(FITS)
    widths = pd.read_csv(WIDTHS)
    cell_ll = pd.read_csv(CELL_LL)
    values = parameter_frames(fits, widths)
    results = {name: paired_parameter(name, values[name].dropna())
               for name in ['field_width', 'peak', 'tau', 'tau_unbound']}
    paired_model, model_result = model_contrast(cell_ll)
    rows = [{key: value for key, value in result.items() if key != 'paired'} for result in results.values()]
    rows.append(model_result)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / 'pooled_tic_comparison.csv', index=False)
    paired_model.to_csv(OUTPUT / 'pooled_tic_model_session_pairs.csv', index=False)
    lines = [
        '# pooled TIC comparison',
        '',
        'TICs combine the existing ramp-down and ramp-up labels. Width uses the independent 0.5 SD threshold table, peak uses the fitted Gaussian peak, and tau uses decay tau for ramp-down and time cells and rise tau for ramp-up. Each p value is a paired Wilcoxon test on the fifteen session medians.',
        '',
        summary.to_string(index=False),
        '',
        'The pooled held-out model contrast is Gaussian log likelihood minus the expected exponential log likelihood: decay for ramp-down TICs, rise for ramp-up TICs, and the better of decay and rise for time cells. Negative values favour the exponential. The session-level test is the relevant inferential result; a cell-level rank test would treat cells from the same session as independent.',
        '',
        paired_model.to_string(index=False),
    ]
    (OUTPUT / 'pooled_tic_comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    make_figure(values, results)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
