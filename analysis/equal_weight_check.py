# -*- coding: utf-8 -*-
'''
Created on Mon Nov 16 13:12:22 2026

check how equal time-bin weights change the saved field parameters

@author: Dinghao Luo
'''

#%% imports
import os
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
import argparse
import json
from itertools import repeat
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.rf_models_v1_2 import BOUNDS, compressed_observations, fit_field, _GRID_CACHE
from analysis.two_exponential import markdown

OUTPUT = ROOT / 'outputs' / 'equal_weight_check'
WIDER  = ROOT / 'outputs' / 'undecided_check'

#%% equal-weight fits
def retained_coordinates(name, t, interval, y, trial=None):
    if name == 'rise':
        # count distinct trials even if two observations land in the same bin
        coordinate = np.floor((interval - t) / .05 + 1e-8).astype(int)
        centres, inverse = np.unique(coordinate, return_inverse=True)
        counts = np.bincount(inverse)
        mean = np.bincount(inverse, weights=y) / counts
        pairs = np.unique(np.column_stack([inverse, trial]), axis=0)
        support = np.bincount(pairs[:, 0], minlength=len(centres))
        keep = support >= 10
        return np.zeros(keep.sum()), (centres[keep] + .5) * .05, mean[keep]
    t, interval, y, counts = compressed_observations(name, t, interval, y)
    keep = counts >= 10
    return t[keep], interval[keep], y[keep]


def session_job(path, rise_only=False):
    started = time.perf_counter()
    old = pd.read_csv(path)
    session = old.session.iloc[0]
    data = np.load(ROOT / 'data' / 'binned_v1_3' / f'{session}.npz')
    rows, failures = [], []
    indices = {int(neuron): i for i, neuron in enumerate(data['neu_id'])}
    for cell in old.itertuples():
        signal = data['dff'][indices[int(cell.neu_id)]]
        trial, bins = np.where(np.isfinite(signal))
        for name in (['rise'] if rise_only else ['gaussian', 'decay', 'rise']):
            t, interval, y = retained_coordinates(name, data['t'][bins], data['T_k'][trial], signal[trial, bins], trial)
            if len(y) < 4:
                failures.append(dict(session=session, neu_id=cell.neu_id, model=name, issue='fewer than four coordinates with 10 trials'))
                continue
            params, ok = fit_field(name, t, interval, y, equal_time_weight=True)
            if not ok:
                failures.append(dict(session=session, neu_id=cell.neu_id, model=name, issue='least_squares did not converge'))
            keys = ['a0', 'a', 'mu', 'sigma' if name == 'gaussian' else 'tau']
            lower, upper = BOUNDS[name]
            for i, (key, value) in enumerate(zip(keys, params)):
                original = getattr(cell, f'{name}_{key}')
                old_bound = getattr(cell, f'{name}_{key}_at_bound') if key != 'a0' else np.nan
                bound = (bool(np.isclose(value, lower[i-2], rtol=0, atol=1e-5) or
                              np.isclose(value, upper[i-2], rtol=0, atol=1e-5)) if i >= 2
                         else bool(value <= 1e-5) if key == 'a' else np.nan)
                rows.append(dict(session=session, neu_id=cell.neu_id, group=cell.group, model=name,
                                 parameter=key, original=original, equal_weight=value,
                                 original_at_bound=old_bound, equal_at_bound=bound,
                                 retained_bins=len(y), fit_success=bool(ok)))
    fitted = pd.DataFrame(rows)
    if rise_only:
        previous = pd.read_csv(OUTPUT / f'{session}_parameters.csv', float_precision='round_trip')
        fitted = pd.concat([previous[previous.model != 'rise'], fitted], ignore_index=True)
        previous_status = json.loads((OUTPUT / f'{session}_status.json').read_text())
        failures = [f for f in previous_status['failures'] if f['model'] != 'rise'] + failures
    fitted.to_csv(OUTPUT / f'{session}_parameters.csv', index=False)
    status = dict(session=session, seconds=time.perf_counter()-started, failures=failures)
    (OUTPUT / f'{session}_status.json').write_text(json.dumps(status,indent=2))
    print(session, 'equal weights finished',round(status['seconds'],1),flush=True)
    _GRID_CACHE.clear()
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rise-only', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        statuses = list(pool.map(session_job, sorted(WIDER.glob('*_wider_parameters.csv')), repeat(args.rise_only)))
    summarise(statuses, time.perf_counter() - started, args.rise_only)


def summarise(statuses, seconds, rise_only=False):
    cells = pd.concat([pd.read_csv(p, float_precision='round_trip') for p in sorted(OUTPUT.glob('*_parameters.csv'))],ignore_index=True)
    if rise_only:
        previous = pd.read_csv(OUTPUT / 'parameters.csv', float_precision='round_trip')
        cells = pd.concat([previous[previous.model != 'rise'], cells[cells.model == 'rise']], ignore_index=True)
    cells.to_csv(OUTPUT / 'parameters.csv', index=False)
    rows = []
    for (group, model, parameter), part in cells.groupby(['group','model','parameter']):
        old_pct = 100 * pd.to_numeric(part.original_at_bound).mean()
        new_pct = 100 * pd.to_numeric(part.equal_at_bound).mean()
        rows.append(dict(group=group, model=model, parameter=parameter, n=len(part),
                         original_median=part.original.median(), equal_median=part.equal_weight.median(),
                         median_absolute_difference=(part.original-part.equal_weight).abs().median(),
                         spearman_r=spearmanr(part.original,part.equal_weight).statistic,
                         original_bound_pct=old_pct, equal_bound_pct=new_pct, bound_change_pp=new_pct-old_pct))
    summary = pd.DataFrame(rows)
    if rise_only:
        previous = pd.read_csv(OUTPUT / 'comparison_summary.csv', float_precision='round_trip')
        summary = pd.concat([previous[previous.model != 'rise'], summary[summary.model == 'rise']], ignore_index=True)
        summary = summary.sort_values(['group', 'model', 'parameter']).reset_index(drop=True)
    retained = cells[(cells.model == 'rise') & (cells.parameter == 'a0')].groupby('session').retained_bins.agg(
        cells='size', minimum='min', median='median', maximum='max').reset_index()
    retained.to_csv(OUTPUT / 'rise_retained_bins.csv', index=False)
    summary.to_csv(OUTPUT / 'comparison_summary.csv',index=False)
    failures = [f for status in statuses for f in status['failures']]
    pd.DataFrame(failures,columns=['session','neu_id','model','issue']).to_csv(OUTPUT / 'failures.csv',index=False)
    text = '# equal time-bin weights\n\nThe full-data Gaussian and decay fits give each retained exact model-axis coordinate equal weight, after removal of coordinates represented in fewer than 10 trials. The rise refits instead average observations within 50 ms bins of T minus t, give each bin equal weight, and retain bins supported by at least 10 distinct trials. Bins are left-closed, right-open, with edges at integer multiples of 50 ms; predictions are evaluated at bin centres. The default count-weighted fit is unchanged. All three models are shown for each group, including the group\'s non-primary exponential. Baseline a0 is unbounded, so its bound percentages are undefined.\n\n'
    unsupported = sum(f['issue'].startswith('fewer than') for f in failures)
    text += '## retained rise bins per session\n\n' + markdown(retained) + '\n\n## parameter comparisons\n\n'
    run_scope = 'rise refits only' if rise_only else 'all models'
    text += markdown(summary) + f'\n\nUnsupported fits: {unsupported}; attempted fits that did not converge: {len(failures) - unsupported}. Wall time for {run_scope}: {seconds:.1f} s. Unsupported fits are listed in failures.csv, and parameter summaries use the defined matched subset. These paired descriptive checks combine equal weighting with the requested 10-trial support restriction.\n'
    if failures:
        counts = pd.DataFrame(failures).groupby(['session', 'model', 'issue']).size().rename('n').reset_index()
        text += '\n' + markdown(counts) + '\n'
    (OUTPUT / 'summary.md').write_text(text,encoding='utf-8')
    (OUTPUT / 'run_report.json').write_text(json.dumps(dict(seconds=seconds,unsupported=unsupported,convergence_failures=len(failures)-unsupported,attempted_fits=cells[cells.parameter == 'a0'].shape[0] if not rise_only else cells[(cells.model == 'rise') & (cells.parameter == 'a0')].shape[0],models_refitted=['rise'] if rise_only else ['gaussian','decay','rise']),indent=2))
    print(text,flush=True)


if __name__ == '__main__':
    main()
