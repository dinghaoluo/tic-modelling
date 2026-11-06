'''
2026-11-06
assemble the reviewer evidence audit
@author: Dinghao Luo
'''

#%% imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'reviewer_audit'
FIGURES = ROOT / 'figures' / 'reviewer_audit'
GROUPS = ['tic_down', 'tic_up', 'time_cell']
DECISIONS = ['gaussian', 'decay', 'rise', 'undecided', 'flat']
COLOURS = {'gaussian': '#4477AA', 'decay': '#CC6677', 'rise': '#228833',
           'undecided': '#CCB84A', 'flat': '#999999'}


#%% summaries

def decision_summary(fits):
    part = fits.loc[fits.group.isin(GROUPS), ['group', 'preferred']].copy()
    part['decision'] = part.preferred.where(part.preferred.isin(DECISIONS), 'undecided')
    table = pd.crosstab(part.group, part.decision, normalize='index').mul(100)
    table = table.reindex(index=GROUPS, columns=DECISIONS, fill_value=0)
    return table.reset_index().rename(columns={'group': 'neuron_group'})


def recovery_summary(recovery):
    wanted = [('tic_down', 'gaussian'), ('tic_down', 'decay'),
              ('tic_up', 'gaussian'), ('tic_up', 'rise'),
              ('time_cell', 'gaussian')]
    rows = []
    for group, shape in wanted:
        part = recovery[(recovery.subset == 'all_sampled') &
                        (recovery.group == group) &
                        (recovery.truth_shape == shape) &
                        (recovery.outcome == 'undecided')]
        if len(part) != 1:
            raise ValueError(f'missing recovery row for {group} {shape}')
        row = part.iloc[0]
        rows.append({'group': group, 'truth_shape': shape, 'n_cells': int(row.n_cells),
                     'undecided_pct': row.percentage, 'lower_pct': row.lower,
                     'upper_pct': row.upper})
    return pd.DataFrame(rows)


def parameter_summary(fits):
    requested = [('time_cell', 'gaussian', 'gaussian_sigma'),
                 ('tic_down', 'decay', 'decay_tau'),
                 ('tic_up', 'rise', 'rise_tau')]
    rows = []
    for group, shape, parameter in requested:
        bound = f'{parameter}_at_bound'
        part = fits[(fits.group == group) & fits.beats_flat & fits[parameter].notna()]
        usable = part.loc[~part[bound], parameter]
        rows.append({
            'group': group,
            'shape': shape,
            'parameter': parameter,
            'n_beats_flat': len(part),
            'pct_at_bound': 100 * part[bound].mean(),
            'n_not_at_bound': len(usable),
            'median_s': usable.median(),
            'iqr_s': usable.quantile(.75) - usable.quantile(.25),
            'sd_s': usable.std(ddof=1),
        })
    return pd.DataFrame(rows)


def scaling_summary(fits):
    part = fits[(fits.group.isin(GROUPS)) & fits.beats_flat & fits.scaled_shape.notna()].copy()
    rows = []
    for (group, shape), cells in part.groupby(['group', 'scaled_shape'], sort=True):
        usable = cells.loc[~cells.rho_at_bound & cells.rho.notna(), 'rho']
        rows.append({
            'group': group,
            'shape': shape,
            'n_cells': len(cells),
            'n_rho_not_at_bound': len(usable),
            'rho_median': usable.median(),
            'rho_iqr': usable.quantile(.75) - usable.quantile(.25),
            'rho_sd': usable.std(ddof=1),
            'pct_gain_interval_above_zero': 100 * (cells.scaling_mse_gain_lower > 0).mean(),
            'pct_rho_at_bound': 100 * cells.rho_at_bound.mean(),
        })
    return pd.DataFrame(rows)


#%% figure

def make_figure(decisions, recovery, fits, widths):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    bottom = np.zeros(len(GROUPS))
    for decision in DECISIONS:
        values = decisions.set_index('neuron_group').loc[GROUPS, decision].to_numpy()
        ax.bar(GROUPS, values, bottom=bottom, color=COLOURS[decision], label=decision)
        bottom += values
    ax.set(ylabel='cells (%)', title='cell-level decisions')
    ax.legend(frameon=False, ncol=3, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[0, 1]
    x = np.arange(len(recovery))
    values = recovery.undecided_pct.to_numpy()
    lower = values - recovery.lower_pct.to_numpy()
    upper = recovery.upper_pct.to_numpy() - values
    colours = [COLOURS['gaussian'] if shape == 'gaussian' else COLOURS['decay'] if shape == 'decay' else COLOURS['rise']
               for shape in recovery.truth_shape]
    for index, (value, low, high, colour) in enumerate(zip(values, lower, upper, colours)):
        ax.errorbar(index, value, yerr=[[low], [high]], fmt='o', color=colour,
                    ecolor=colour, elinewidth=2, capsize=3)
    ax.set_xticks(x, [f'{g.replace("tic_", "")} {s}' for g, s in zip(recovery.group, recovery.truth_shape)],
                  rotation=35, ha='right')
    ax.set(ylabel='undecided (%)', title='recovery with known truth')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1, 0]
    requested = [('time_cell', 'gaussian_sigma', '#999999'),
                 ('tic_down', 'decay_tau', '#4477AA'),
                 ('tic_up', 'rise_tau', '#CC6677')]
    box_values = []
    labels = []
    for group, parameter, colour in requested:
        bound = f'{parameter}_at_bound'
        part = fits[(fits.group == group) & fits.beats_flat & ~fits[bound] & fits[parameter].notna()]
        box_values.append(part[parameter].to_numpy())
        labels.append(f'{group.replace("_", " ")}\n{parameter.replace("_", " ")}')
    artists = ax.boxplot(box_values, tick_labels=labels, showfliers=False, patch_artist=True)
    for artist, colour in zip(artists['boxes'], [item[2] for item in requested]):
        artist.set_facecolor(colour)
        artist.set_alpha(.55)
    ax.set(ylabel='parameter (s)', title='supported, non-bound fits')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1, 1]
    groups = ['tic_down', 'tic_up', 'tic_and_time_cell', 'time_cell']
    colours = ['#4477AA', '#CC6677', '#228833', '#999999']
    values = []
    labels = []
    censored = []
    for group in groups:
        part = widths[(widths.neuron_type == group) & (widths.subset == 'all')]
        values.append(part.width.to_numpy())
        labels.append(group.replace('_', ' '))
        censored.append(part.censored.to_numpy())
    artists = ax.boxplot(values, tick_labels=labels, showfliers=False, patch_artist=True)
    for artist, colour in zip(artists['boxes'], colours):
        artist.set_facecolor(colour)
        artist.set_alpha(.55)
    for index, (value, mask) in enumerate(zip(values, censored), start=1):
        if mask.any():
            ax.scatter(np.full(mask.sum(), index), value[mask], color='black', marker='^', s=16, zorder=3)
    ax.axhline(6.99, color='black', linestyle=':', linewidth=.7)
    ax.set(ylabel='threshold width (s)', title='independent threshold widths')
    ax.tick_params(axis='x', rotation=35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('receptive-field evidence audit', y=.995)
    fig.tight_layout()
    fig.savefig(FIGURES / 'reviewer_evidence.png', dpi=300)
    plt.close(fig)


#%% run

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    fits = pd.read_csv(ROOT / 'outputs' / 'fits_v1_2' / 'fits_all.csv')
    recovery = pd.read_csv(ROOT / 'outputs' / 'undecided_check' / 'recovery_summary.csv')
    widths = pd.read_csv(ROOT / 'outputs' / 'width_summary' / 'widths.csv')

    decisions = decision_summary(fits)
    recovery_table = recovery_summary(recovery)
    parameters = parameter_summary(fits)
    scaling = scaling_summary(fits)

    decisions.to_csv(OUT / 'decision_rates.csv', index=False)
    recovery_table.to_csv(OUT / 'recovery_undecided.csv', index=False)
    parameters.to_csv(OUT / 'parameter_summary.csv', index=False)
    scaling.to_csv(OUT / 'scaling_summary.csv', index=False)
    widths[widths.subset == 'all'].to_csv(OUT / 'width_summary_all.csv', index=False)

    report = '\n'.join([
        '# reviewer evidence audit',
        '',
        'The cell-level decision rule is reported with flat and undecided cells retained. Recovery percentages show the expected undecided rate when the generating shape is known. Parameter summaries use cells that beat flat and exclude fits with a bound parameter. Threshold widths are the independent 0.5 SD measurements, with endpoint cases retained and marked as censored.',
        '',
        '## decision rates',
        decisions.to_string(index=False),
        '',
        '## recovery undecided rates',
        recovery_table.to_string(index=False),
        '',
        '## supported parameter summaries',
        parameters.to_string(index=False),
        '',
        '## interval scaling summaries',
        scaling.to_string(index=False),
    ])
    (OUT / 'summary.md').write_text(report, encoding='utf-8')
    make_figure(decisions, recovery_table, fits, widths)


if __name__ == '__main__':
    main()