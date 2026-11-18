'''
Created on Wed Nov 18 00:39:00 2026

illustrate onset smoothing and trial intervals in the saved fits

@author: Dinghao Luo
'''

#%% imports
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.fit_quality import average_traces
from analysis.revised_two_exponential import predict_model
from utils.rf_models import predict

SESSION = 'A133-20230114-02'
NEU_ID  = 88
FIGURES = ROOT / 'figures'

#%% figure
def make_figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parameters = pd.read_csv(ROOT / 'outputs' / 'wider_window_fits' / f'{SESSION}_wider_parameters.csv')
    selected = parameters[parameters.neu_id.eq(NEU_ID)]
    assert len(selected) == 1
    cell = next(selected.itertuples())
    assert cell.group == 'tic_down'
    revised = pd.read_csv(ROOT / 'outputs' / 'revised_two_exponential' / 'parameters.csv')
    selected = revised[revised.session.eq(SESSION) & revised.neu_id.eq(NEU_ID)
                       & revised.model.eq('revised_last')]
    assert len(selected) == 1
    revised_params = selected.iloc[0][['a', 'B', 't0', 'u', 'tau1', 'tau2']].to_numpy(float)
    with np.load(ROOT / 'data' / 'binned_wider' / f'{SESSION}.npz') as data:
        times, intervals, ids = data['t'], data['T_k'], data['neu_id']
        assert np.count_nonzero(ids == NEU_ID) == 1
        signal = data['dff'][np.flatnonzero(ids == NEU_ID)[0]].astype(float)
    assert signal.shape == (len(intervals), len(times))
    observed = np.isfinite(signal)
    assert not np.any(observed & (times[None, :] > intervals[:, None]))
    np.testing.assert_allclose(np.diff(times), .05, rtol=0, atol=1e-10)

    exponential_params = np.array([cell.decay_a0, cell.decay_a, cell.decay_mu, cell.decay_tau])
    curve_times = np.linspace(-1., intervals.max(), 1500)
    exponential = predict('decay', curve_times, np.zeros_like(curve_times), exponential_params)
    baseline, amplitude, onset, tau = exponential_params
    without_onset = baseline + amplitude * np.exp(-np.maximum(curve_times - onset, 0.) / tau)
    trial_predictions = predict_model('revised_last', times[None, :], intervals[:, None], revised_params)
    trace_times, _, _, _, mean_prediction = average_traces(
        signal, times, intervals, cell, 'last_lick', additional=[trial_predictions])

    plt.rcParams.update({'font.size': 16, 'axes.labelsize': 16,
                         'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'pdf.use14corefonts': False})
    figure, axes = plt.subplots(1, 2, figsize=(13., 5.2), sharex=True, sharey=True)
    figure.subplots_adjust(left=.08, right=.985, bottom=.18, top=.76, wspace=.24)
    axes[0].plot(curve_times, without_onset, color='#999999', linewidth=2.,
                 label='Same parameters, onset removed')
    axes[0].plot(curve_times, exponential, color='black', linestyle='--', linewidth=2.,
                 label='Fitted exponential')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[::-1], labels[::-1], loc='lower left', bbox_to_anchor=(0., 1.03),
                   frameon=False, borderaxespad=0.)

    for trial, interval in enumerate(intervals):
        trial_times = np.r_[times[observed[trial]], interval]
        prediction = predict_model('revised_last', trial_times, interval, revised_params)
        axes[1].plot(trial_times, prediction, color='#BBBBBB', linewidth=.65,
                     label='Individual trials' if trial == 0 else None)
    axes[1].plot(trace_times, mean_prediction, color='#0072B2', linestyle='-.', linewidth=2.5,
                 label='Mean across observed trials')
    axes[1].legend(loc='lower left', bbox_to_anchor=(0., 1.03), frameon=False, borderaxespad=0.)
    for axis in axes:
        axis.set(xlabel='Time from last lick (s)', ylabel='dF/F', xlim=(-1., intervals.max()))
        axis.spines[['top', 'right']].set_visible(False)
        axis.tick_params(axis='y', labelleft=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    destination = FIGURES / 'fit_explanation'
    figure.savefig(destination.with_suffix('.png'), dpi=300)
    figure.savefig(destination.with_suffix('.pdf'))
    plt.close(figure)
    print(f'{SESSION}, neuron {NEU_ID}: {len(intervals)} trial curves; {len(trace_times)} averaged bins')


if __name__ == '__main__':
    make_figure()
