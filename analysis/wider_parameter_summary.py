'''
2026-11-07
summarise wider-window parameters and the aligned Cao comparison
@author: Dinghao Luo
'''

#%% imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
WIDER = ROOT / 'outputs' / 'undecided_check'
WIDTHS = ROOT / 'outputs' / 'width_summary' / 'widths.csv'
OUTPUT = ROOT / 'outputs' / 'wider_evidence'
FIGURES = ROOT / 'figures' / 'wider_evidence'
CAO = {
    'tic_down': {'label': 'past', 'mu': .01, 'mu_q25': 0., 'mu_q75': .10, 'tau': 2.25, 'tau_q25': .59, 'tau_q75': 4.},
    'tic_up': {'label': 'future', 'mu': -.02, 'mu_q25': -.13, 'mu_q75': -.01, 'tau': 1.03, 'tau_q25': .26, 'tau_q75': 3.},
}


def markdown(frame):
    columns = list(frame.columns)
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    lines.extend('| ' + ' | '.join(str(value) for value in row) + ' |'
                 for row in frame.itertuples(index=False, name=None))
    return '\n'.join(lines)


def distribution(values):
    values = pd.Series(values).dropna()
    return dict(n=len(values), median=values.median(), q25=values.quantile(.25), q75=values.quantile(.75))


def paired(values, name):
    pairs = []
    for session, part in values.groupby('session'):
        tic = part.loc[part.group.eq('tic'), name].dropna()
        time = part.loc[part.group.eq('time'), name].dropna()
        if len(tic) and len(time):
            pairs.append(dict(session=session, tic_median=tic.median(), time_median=time.median(),
                              difference=tic.median() - time.median()))
    pairs = pd.DataFrame(pairs)
    result = wilcoxon(pairs.difference) if len(pairs) else None
    return dict(quantity=name, n_sessions=len(pairs), pooled_tic=values.loc[values.group.eq('tic'), name].median(),
                pooled_time=values.loc[values.group.eq('time'), name].median(),
                session_difference_median=pairs.difference.median(),
                sessions_tic_larger=int((pairs.difference > 0).sum()),
                p_wilcoxon=result.pvalue if result is not None else np.nan), pairs


def load_parameters():
    parameters = pd.concat([pd.read_csv(path) for path in WIDER.glob('*_wider_parameters.csv')], ignore_index=True)
    original = parameters.group.copy()
    parameters['group'] = np.where(original.isin(['tic_down', 'tic_up']), 'tic',
                                   np.where(original.eq('time_cell'), 'time', 'other'))
    parameters['peak'] = parameters.gaussian_mu.where(~parameters.gaussian_mu_at_bound)
    parameters['tau'] = np.where(original.eq('tic_up'), parameters.rise_tau, parameters.decay_tau)
    tau_bound = np.where(original.eq('tic_up'), parameters.rise_tau_at_bound, parameters.decay_tau_at_bound).astype(bool)
    parameters['tau_unbound'] = parameters.tau.where(~tau_bound)
    return parameters


def load_widths():
    widths = pd.read_csv(WIDTHS)
    widths = widths[widths.subset.eq('all')].copy()
    widths['group'] = np.where(widths.neuron_type.isin(['tic_down', 'tic_up']), 'tic',
                               np.where(widths.neuron_type.eq('time_cell'), 'time', 'other'))
    return widths.rename(columns={'width': 'field_width'})[['session', 'group', 'field_width']]


def parameter_table(parameters, widths):
    values = {
        'field_width': widths,
        'peak': parameters[['session', 'group', 'peak']],
        'tau': parameters[['session', 'group', 'tau']],
        'tau_unbound': parameters[['session', 'group', 'tau_unbound']],
    }
    rows, pairs = [], []
    for name, frame in values.items():
        result, pair = paired(frame, name)
        rows.append(result)
        pair['quantity'] = name
        pairs.append(pair)
    return pd.DataFrame(rows), pd.concat(pairs, ignore_index=True)


def cao_table(parameters):
    rows = []
    for group, shape in [('tic_down', 'decay'), ('tic_up', 'rise')]:
        part = parameters[parameters.original_group.eq(group)] if 'original_group' in parameters else None
        if part is None:
            raise ValueError('original group column missing')
        tau = part[f'{shape}_tau']
        bound = part[f'{shape}_tau_at_bound'].astype(bool)
        for subset, kept in [('all', tau), ('tau_bound_excluded', tau[~bound])]:
            stats = distribution(kept)
            rows.append(dict(quantity='tau', group=group, subset=subset, n=stats['n'], median=stats['median'],
                             q25=stats['q25'], q75=stats['q75'], bound_pct=100 * bound.mean(),
                             cao_median=CAO[group]['tau'], cao_q25=CAO[group]['tau_q25'], cao_q75=CAO[group]['tau_q75']))
        mu = part.gaussian_mu[~part.gaussian_mu_at_bound]
        stats = distribution(mu)
        rows.append(dict(quantity='peak_mu', group=group, subset='gaussian_mu_bound_excluded', n=stats['n'],
                         median=stats['median'], q25=stats['q25'], q75=stats['q75'], bound_pct=100 * part.gaussian_mu_at_bound.mean(),
                         cao_median=CAO[group]['mu'], cao_q25=CAO[group]['mu_q25'], cao_q75=CAO[group]['mu_q75']))
    return pd.DataFrame(rows)


def model_summary():
    contrasts = pd.concat([pd.read_csv(path) for path in WIDER.glob('*_wider_contrasts.csv')], ignore_index=True)
    primary = contrasts[contrasts.primary]
    rows = []
    for group, part in primary.groupby('group'):
        rows.append(dict(group=group, n_cells=len(part), n_sessions=part.session.nunique(),
                         gaussian_better=100 * part.decision.eq('gaussian').mean(),
                         exponential_better=100 * part.decision.isin(['decay', 'rise']).mean(),
                         undecided=100 * part.decision.eq('undecided').mean(),
                         median_contrast=part.mean_contrast.median()))
    return pd.DataFrame(rows)


def figure(parameters, widths, table):
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'pdf.use14corefonts': False})
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    values = {'field_width': widths, 'peak': parameters[['session', 'group', 'peak']],
              'tau_unbound': parameters[['session', 'group', 'tau_unbound']]}
    labels = {'field_width': '0.5 SD width (s)', 'peak': 'Gaussian peak (s)',
              'tau_unbound': 'matched tau (bound excluded, s)'}
    colours = {'tic': '#4477AA', 'time': '#AA55AA'}
    for axis, name in zip(axes, ['field_width', 'peak', 'tau_unbound']):
        for group in ['tic', 'time']:
            part = values[name].loc[values[name].group.eq(group), name].dropna()
            axis.hist(part, bins=20, density=True, alpha=.6, color=colours[group],
                      label='TICs' if group == 'tic' else 'time cells')
        p = table.loc[table.quantity.eq(name), 'p_wilcoxon'].iloc[0]
        axis.set(xlabel=labels[name], ylabel='density', title=f'p = {p:.3g}')
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('TIC and time-cell parameter distributions')
    fig.tight_layout()
    fig.savefig(FIGURES / 'parameter_distributions.png', dpi=300)
    fig.savefig(FIGURES / 'parameter_distributions.pdf')
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    parameters = load_parameters()
    parameters['original_group'] = np.where(parameters.group.eq('tic'), 'tic',
                                            np.where(parameters.group.eq('time'), 'time', 'other'))
    # recover the directional labels needed for the Cao table from the source files
    source = pd.concat([pd.read_csv(path) for path in WIDER.glob('*_wider_parameters.csv')], ignore_index=True)
    parameters['original_group'] = source.group.to_numpy()
    widths = load_widths()
    table, pairs = parameter_table(parameters, widths)
    cao = cao_table(parameters)
    models = model_summary()
    table.to_csv(OUTPUT / 'wider_parameter_summary.csv', index=False)
    pairs.to_csv(OUTPUT / 'wider_parameter_session_pairs.csv', index=False)
    cao.to_csv(OUTPUT / 'wider_cao_summary.csv', index=False)
    models.to_csv(OUTPUT / 'wider_model_summary.csv', index=False)
    report = ['# parameter distribution summary', '',
              'All model parameters and model comparisons in this report come from the analysis interval used for the fitted traces. Width is the independent 0.5 SD threshold measurement. Peak and tau distributions report bound-excluded summaries where indicated; the all-fit tau rows retain the upper-bound values explicitly.', '',
              '## pooled parameters', markdown(table), '',
              '## Cao comparison', markdown(cao), '',
              '## held-out model comparison', markdown(models), '',
              'The held-out percentages are the wider-window Gaussian-versus-directional-exponential decisions. They retain undecided cells and should not be read as a claim that most cells have a decisive model identity.']
    (OUTPUT / 'wider_parameter_summary.md').write_text('\n'.join(report), encoding='utf-8')
    figure(parameters, widths, table)
    print(table.to_string(index=False))
    print(cao.to_string(index=False))
    print(models.to_string(index=False))


if __name__ == '__main__':
    main()

