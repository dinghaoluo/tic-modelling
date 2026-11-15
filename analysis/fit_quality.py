'''Created on 15 November 2026
compare saved fits with trial-averaged activity
@author: Dinghao Luo
'''

#%% imports
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.rf_models_v1_2 import predict

DATA       = ROOT / 'data' / 'binned_v1_3'
PARAMETERS = ROOT / 'outputs' / 'undecided_check'
OUTPUT     = ROOT / 'outputs' / 'fit_quality'
FIGURES    = ROOT / 'figures'
MIN_TRIALS = 10
GROUPS     = ['tic_down', 'tic_up', 'time_cell']
LABELS     = ['Ramp-down TIC', 'Ramp-up TIC', 'Time cells']
COLOURS    = ['#228833', '#4477AA', '#9467BD']


#%% averaged traces
def r_squared(observed, prediction):
    '''score shared finite bins against their observed mean, retaining negative R²'''
    observed = np.asarray(observed, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if observed.shape != prediction.shape:
        raise ValueError('observed and predicted traces must have the same shape')
    keep = np.isfinite(observed) & np.isfinite(prediction)
    if keep.sum() < 2:
        return np.nan
    y, fitted = observed[keep], prediction[keep]
    total = np.sum((y - y.mean()) ** 2)
    if total == 0:
        return np.nan
    return float(1. - np.sum((y - fitted) ** 2) / total)


def average_traces(signal, times, intervals, cell, alignment):
    '''apply the same observation mask and alignment operator to data and both fits'''
    exponential = 'rise' if cell.group == 'tic_up' else 'decay'
    mask = np.isfinite(signal)
    traces = [signal]
    for name in ['gaussian', exponential]:
        width = 'sigma' if name == 'gaussian' else 'tau'
        params = np.array([getattr(cell, f'{name}_{key}') for key in ['a0', 'a', 'mu', width]])
        fitted = np.broadcast_to(predict(name, times[None, :], intervals[:, None], params), signal.shape)
        if not np.isfinite(fitted[mask]).all():
            raise ValueError(f'{cell.session}, neuron {cell.neu_id}: non-finite prediction')
        traces.append(np.where(mask, fitted, np.nan))
    if alignment == 'next_lick':
        step = float(np.median(np.diff(times)))
        axis = np.arange(int(np.ceil((intervals.max() - times.min()) / step)) + 1) * step
        aligned = [np.full((len(intervals), len(axis)), np.nan) for _ in traces]
        for trial, interval in enumerate(intervals):
            query = interval - axis
            right = np.searchsorted(times, query, side='right')
            inside = (query >= times[0]) & (query <= times[-1])
            right = np.clip(right, 1, len(times) - 1)
            left = right - 1
            weight = (query - times[left]) / (times[right] - times[left])
            # missing endpoints cannot supply interpolated activity
            supported = inside & mask[trial, left] & mask[trial, right]
            supported |= inside & (weight == 0.) & mask[trial, left]
            supported |= inside & (weight == 1.) & mask[trial, right]
            for values, result in zip(traces, aligned):
                y0, y1 = values[trial, left], values[trial, right]
                interpolated = np.where(weight == 0., y0,
                                        np.where(weight == 1., y1, (1. - weight) * y0 + weight * y1))
                result[trial, supported] = interpolated[supported]
        traces = aligned
        mask = np.isfinite(traces[0])
    else:
        axis = times
    counts = mask.sum(axis=0)
    keep = counts >= MIN_TRIALS
    means = [np.nansum(values, axis=0)[keep] / counts[keep] for values in traces]
    return axis[keep], *means


#%% saved-parameter comparison
def compute_cells():
    rows, curves, issues = [], {}, []
    for path in sorted(PARAMETERS.glob('*_wider_parameters.csv')):
        parameters = pd.read_csv(path)
        session = parameters.session.iloc[0]
        with np.load(DATA / f'{session}.npz') as data:
            signals, times, intervals, ids = [data[key] for key in ['dff', 't', 'T_k', 'neu_id']]
        assert signals.shape == (len(ids), len(intervals), len(times))
        assert len(set(ids)) == len(ids) and not parameters.neu_id.duplicated().any()
        assert set(ids) == set(parameters.neu_id)
        assert np.all(np.diff(times) > 0) and np.isfinite(intervals).all()
        np.testing.assert_allclose(np.diff(times), .05, rtol=0, atol=1e-10)
        indices = {int(neuron): i for i, neuron in enumerate(ids)}
        for cell in parameters.itertuples():
            if cell.group not in GROUPS:
                issues.append(f'{session}, neuron {cell.neu_id}: unexpected group {cell.group}, omitted')
                continue
            signal = signals[indices[int(cell.neu_id)]].astype(float)
            observed = np.isfinite(signal)
            assert not np.any(observed & (times[None, :] > intervals[:, None]))
            if np.any(np.diff(observed.astype(int), axis=1) > 0):
                issues.append(f'{session}, neuron {cell.neu_id}: missing bins within the observed window')
            if not cell.fit_success:
                issues.append(f'{session}, neuron {cell.neu_id} ({cell.group}): saved fit_success=False; retained')
            for alignment in ['last_lick', 'next_lick'] if cell.group == 'tic_up' else ['last_lick']:
                trace = average_traces(signal, times, intervals, cell, alignment)
                r2_gaussian = r_squared(trace[1], trace[2])
                r2_exponential = r_squared(trace[1], trace[3])
                rows.append(dict(session=session, neu_id=int(cell.neu_id), group=cell.group,
                                 r2_gaussian=r2_gaussian, r2_exponential=r2_exponential, alignment=alignment))
                curves[(session, int(cell.neu_id), alignment)] = trace
                if not np.isfinite([r2_gaussian, r2_exponential]).all():
                    issues.append(f'{session}, neuron {cell.neu_id}, {alignment}: undefined R² ({len(trace[0])} retained bins)')
        print(session, len(parameters), 'cells averaged', flush=True)
    return pd.DataFrame(rows), curves, issues


def summary_table(cells):
    rows = []
    for group, label in zip(GROUPS, LABELS):
        for alignment in ['next_lick', 'last_lick'] if group == 'tic_up' else ['last_lick']:
            part = cells[cells.group.eq(group) & cells.alignment.eq(alignment)]
            finite = np.isfinite(part[['r2_gaussian', 'r2_exponential']]).all(axis=1)
            row = dict(group=label, alignment=alignment, n=len(part), n_defined=int(finite.sum()))
            for model in ['gaussian', 'exponential']:
                values = part.loc[finite, f'r2_{model}']
                row[f'{model}_median'] = values.median()
                row[f'{model}_q25'] = values.quantile(.25)
                row[f'{model}_q75'] = values.quantile(.75)
                row[f'{model}_pct_above_half'] = 100. * (values > .5).mean()
            row['median_difference'] = (part.loc[finite, 'r2_gaussian'] - part.loc[finite, 'r2_exponential']).median()
            rows.append(row)
    return pd.DataFrame(rows)


def choose_examples(cells):
    rows = []
    for group in GROUPS:
        alignment = 'next_lick' if group == 'tic_up' else 'last_lick'
        part = cells[cells.group.eq(group) & cells.alignment.eq(alignment)].copy()
        part = part.dropna(subset=['r2_gaussian', 'r2_exponential'])
        part['larger_r2'] = part[['r2_gaussian', 'r2_exponential']].max(axis=1)
        part = part.sort_values(['larger_r2', 'session', 'neu_id']).reset_index(drop=True)
        for percentile in [20, 40, 60, 80]:
            rank = int(np.rint((len(part) - 1) * percentile / 100.))
            rows.append(dict(percentile=percentile, **part.iloc[rank].to_dict()))
    return pd.DataFrame(rows)


#%% figure
def make_figure(cells, examples, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({'font.size': 16, 'axes.labelsize': 14, 'axes.titlesize': 17,
                         'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'pdf.use14corefonts': False})
    fig = plt.figure(figsize=(13., 14.5))
    grid = fig.add_gridspec(4, 1, height_ratios=[1., 1., 1., 1.15], hspace=.95,
                            left=.08, right=.985, bottom=.06, top=.92)
    bottom_grid = grid[3].subgridspec(1, 3, wspace=.65)
    for row, (group, label, colour) in enumerate(zip(GROUPS, LABELS, COLOURS)):
        selected = examples[examples.group.eq(group)]
        example_grid = grid[row].subgridspec(1, 4, wspace=.75)
        for column, cell in enumerate(selected.itertuples()):
            axis = fig.add_subplot(example_grid[column])
            x, observed, gaussian, exponential = curves[(cell.session, int(cell.neu_id), cell.alignment)]
            axis.plot(x, observed, color=colour, linewidth=2.)
            axis.plot(x, gaussian, color='black', linewidth=1.5)
            axis.plot(x, exponential, color='black', linewidth=1.5, linestyle='--')
            axis.set(xlabel='Time before\nnext lick (s)' if group == 'tic_up' else 'Time from\nlast lick (s)',
                     ylabel='dF/F')
            axis.yaxis.labelpad = 1.
            axis.xaxis.set_major_locator(MaxNLocator(4))
            axis.yaxis.set_major_locator(MaxNLocator(3))
            axis.spines[['top', 'right']].set_visible(False)
            if column == 0:
                fig.text(.08, axis.get_position().y1 + .015, label, fontsize=17, color=colour)
        alignment = 'next_lick' if group == 'tic_up' else 'last_lick'
        part = cells[cells.group.eq(group) & cells.alignment.eq(alignment)]
        axis = fig.add_subplot(bottom_grid[row])
        boxes = axis.boxplot([part.r2_gaussian.dropna(), part.r2_exponential.dropna()],
                             tick_labels=['Gaussian', 'Exponential'], patch_artist=True,
                             medianprops={'color': 'black', 'linewidth': 1.5},
                             flierprops={'markersize': 3, 'markeredgecolor': colour, 'alpha': .5})
        for box in boxes['boxes']:
            box.set(facecolor=colour, alpha=.55)
        axis.axhline(0, color='#999999', linewidth=.8)
        axis.set(title=label, ylabel='Variance explained (R²)')
        axis.spines[['top', 'right']].set_visible(False)
    fig.legend(handles=[Line2D([0], [0], color='#777777', linewidth=2., label='Observed mean (group colour)'),
                        Line2D([0], [0], color='black', linewidth=1.5, label='Gaussian fit'),
                        Line2D([0], [0], color='black', linewidth=1.5, linestyle='--', label='Exponential fit')],
               loc='upper center', bbox_to_anchor=(.5, .985), frameon=False, ncol=3)
    fig.savefig(FIGURES / 'fit_examples.png', dpi=300)
    fig.savefig(FIGURES / 'fit_examples.pdf')
    plt.close(fig)


#%% report
def write_summary(summary, examples, issues, curves):
    lines = ['# fit quality of trial-averaged activity', '',
             'The saved full-data parameters are evaluated on the original trial bins, and predictions inherit each cell’s observed NaN mask before averaging; no parameters are refitted. The exponential is a decay for ramp-down TICs and time cells, and a rise for ramp-up TICs.', '',
             'Last-lick averages use the saved t bins, including the pre-last-lick second. Next-lick averages use time before the next lick (T_k minus t) on a 0.05 s grid starting at zero; observed values and both predictions are linearly interpolated within each trial’s shared observed support, without extrapolation or interpolation across missing endpoints. Every retained bin has at least 10 contributing trials.', '',
             'R² is 1 minus the squared error of the averaged prediction divided by the squared deviation of the averaged observed trace from its mean, with equal weight for each retained time bin; negative values remain negative, while constant traces or fewer than two shared bins give an undefined value. Ramp-up TICs use next-lick alignment as primary. Differences below are Gaussian minus exponential, computed within each cell before taking the median.', '',
             '| Group | Alignment | n (defined) | Gaussian median [IQR] | Exponential median [IQR] | Median difference | Gaussian >0.5 (%) | Exponential >0.5 (%) |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in summary.itertuples():
        lines.append(f'| {row.group} | {row.alignment} | {row.n} ({row.n_defined}) | '
                     f'{row.gaussian_median:.3f} [{row.gaussian_q25:.3f}, {row.gaussian_q75:.3f}] | '
                     f'{row.exponential_median:.3f} [{row.exponential_q25:.3f}, {row.exponential_q75:.3f}] | '
                     f'{row.median_difference:.3f} | {row.gaussian_pct_above_half:.1f} | {row.exponential_pct_above_half:.1f} |')
    lines += ['', 'The IQR describes variation between cells rather than uncertainty in the population median. These are descriptive fits to the same trials used for fitting, so agreement with the mean can be optimistic; the saved held-out trial comparison is the check that separates reproducible shape from fitting the noise.', '',
              '## example cells', '',
              'Within each group’s primary alignment, cells are ordered by the larger of their two R² values, with session and neuron ID breaking ties. The selected zero-based rank is the nearest integer to (n minus 1) times the requested percentile; columns run from the 20th to the 80th percentile. No cells are selected by visual appearance.', '',
              '| Group | Percentile | Session | neu_id | Gaussian R² | Exponential R² | Retained bins |',
              '|---|---:|---|---:|---:|---:|---:|']
    for row in examples.itertuples():
        trace = curves[(row.session, int(row.neu_id), row.alignment)]
        lines.append(f'| {row.group} | {row.percentile} | {row.session} | {int(row.neu_id)} | '
                     f'{row.r2_gaussian:.3f} | {row.r2_exponential:.3f} | {len(trace[0])} |')
    lines += ['', '## input observations', '',
              'The three saved fit_success=False flags correspond to cross-validation fold failures in the saved session status files; none records a failure of the full-data fit used here.', '',
              'The 15 saved sessions contain 404 ramp-down TICs, 479 ramp-up TICs and 1,177 time cells. The saved t bins are 0.05 s apart, running from -0.976 to 6.974 s; neuron IDs are matched explicitly rather than treated as array indices. All retained bins and R² values use the original dF/F units.', '']
    lines += [f'- {issue}' for issue in issues] if issues else ['No discrepancies were found.']
    (OUTPUT / 'fit_quality_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    cells, curves, issues = compute_cells()
    cells.to_csv(OUTPUT / 'fit_quality_cells.csv', index=False)
    summary = summary_table(cells)
    print(summary.to_string(index=False), flush=True)
    print('Input observations:', *issues, sep='\n', flush=True)
    examples = choose_examples(cells)
    examples.to_csv(OUTPUT / 'fit_quality_examples.csv', index=False)
    write_summary(summary, examples, issues, curves)
    make_figure(cells, examples, curves)


if __name__ == '__main__':
    main()
