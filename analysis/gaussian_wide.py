# -*- coding: utf-8 -*-
'''
Created on Mon Nov 16 13:12:22 2026

relax the Gaussian width bound without changing held-out trial scoring

@author: Dinghao Luo
'''

#%% imports
import os
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.rf_models import fit_field, predict, _GRID_CACHE
from utils.trial_error_scores import trial_mse
from analysis.two_exponential import contrast_row, signed_test, markdown

WIDER  = ROOT / 'outputs' / 'wider_window_fits'
OUTPUT = ROOT / 'outputs' / 'gaussian_wide'
BOUNDS = (np.array([-1., .05]), np.array([7., 10.]))

#%% refits
def session_job(path):
    started = time.perf_counter()
    parameters = pd.read_csv(path)
    session = parameters.session.iloc[0]
    data = np.load(ROOT / 'data' / 'binned_wider' / f'{session}.npz')
    fold_data = np.load(ROOT / 'outputs' / 'fits' / f'{session}_folds.npz')
    np.testing.assert_array_equal(data['trial_rows'], fold_data['trial_rows'])
    old_contrasts = pd.read_csv(WIDER / f'{session}_wider_contrasts.csv')
    folds, bootstrap = fold_data['folds'],fold_data['bootstrap_trials']
    indices = {int(neuron):i for i,neuron in enumerate(data['neu_id'])}
    rows, failures = [],[]
    with h5py.File(WIDER / f'{session}_wider_oof.h5','r') as saved:
        np.testing.assert_array_equal(saved['neu_id'][:],data['neu_id'])
        np.testing.assert_array_equal(saved['trial_rows'][:],data['trial_rows']+1)
        for cell in parameters.itertuples():
            index = indices[int(cell.neu_id)]
            signal = data['dff'][index].astype(float)
            held = np.full(signal.shape,np.nan)
            for stage in [0,1,2,3,4,'full']:
                train = np.ones(len(folds),dtype=bool) if stage == 'full' else folds != stage
                trial,bins = np.where(np.isfinite(signal) & train[:,None])
                params,ok = fit_field('gaussian',data['t'][bins],data['T_k'][trial],signal[trial,bins],bounds=BOUNDS)
                if not ok:
                    failures.append(dict(session=session,neu_id=cell.neu_id,stage=str(stage),issue='least_squares did not converge'))
                if stage == 'full':
                    full = params
                else:
                    trial,bins = np.where(np.isfinite(signal) & ~train[:,None])
                    held[trial,bins] = predict('gaussian',data['t'][bins],data['T_k'][trial],params)
            exponential = 'rise' if cell.group == 'tic_up' else 'decay'
            old = old_contrasts[old_contrasts.neu_id.eq(cell.neu_id) & old_contrasts.contrast.eq(f'gaussian_minus_{exponential}')].iloc[0]
            result = contrast_row(trial_mse(signal,held),trial_mse(signal,saved[exponential][index]),old.flat_residual_variance,bootstrap)
            row = dict(session=session,neu_id=cell.neu_id,group=cell.group,**result,
                       old_sigma=cell.gaussian_sigma, old_sigma_at_upper=bool(np.isclose(cell.gaussian_sigma,4.,rtol=0,atol=1e-5)),
                       new_sigma_at_upper=bool(np.isclose(full[3],10.,rtol=0,atol=1e-5)),
                       old_mean_contrast=old.mean_contrast,old_lower=old.lower,old_upper=old.upper,old_decision=old.decision)
            row.update(zip(['a0','a','mu','sigma'],full))
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUTPUT / f'{session}_cells.csv',index=False)
    status=dict(session=session,seconds=time.perf_counter()-started,failures=failures)
    (OUTPUT / f'{session}_status.json').write_text(json.dumps(status,indent=2))
    print(session,'wide Gaussian finished',round(status['seconds'],1),flush=True)
    _GRID_CACHE.clear()
    return status


def main():
    started=time.perf_counter()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        statuses=list(pool.map(session_job,sorted(WIDER.glob('*_wider_parameters.csv'))))
    summarise(statuses, started)


def summarise(statuses, started):
    cells=pd.concat([pd.read_csv(p) for p in sorted(OUTPUT.glob('*_cells.csv'))],ignore_index=True)
    cells.to_csv(OUTPUT / 'cells.csv',index=False)
    rows=[]
    for group,part in cells.groupby('group'):
        for version in ['old','new']:
            decision=part.old_decision.replace({'gaussian':'favour_a','decay':'favour_b','rise':'favour_b'}) if version=='old' else part.decision
            values=part.old_mean_contrast if version=='old' else part.mean_contrast
            medians=values.groupby(part.session).median()
            sigma=part.old_sigma if version=='old' else part.sigma
            rows.append(dict(group=group,bound_s=4 if version=='old' else 10,n=len(part),sessions=len(medians),
                             sigma_median=sigma.median(),sigma_q25=sigma.quantile(.25),sigma_q75=sigma.quantile(.75),
                             at_4_pct=100*np.isclose(sigma,4.,rtol=0,atol=1e-5).mean(),
                             at_10_pct=100*np.isclose(sigma,10.,rtol=0,atol=1e-5).mean(),
                             gaussian_better=100*decision.eq('favour_a').mean(),exponential_better=100*decision.eq('favour_b').mean(),
                             undecided=100*decision.eq('undecided').mean(),median_session_contrast=medians.median(),wilcoxon_p=signed_test(medians)))
    summary=pd.DataFrame(rows)
    summary.to_csv(OUTPUT / 'comparison_summary.csv',index=False)
    failures=[f for status in statuses for f in status['failures']]
    pd.DataFrame(failures,columns=['session','neu_id','stage','issue']).to_csv(OUTPUT / 'failures.csv',index=False)
    seconds=time.perf_counter()-started
    text='# Gaussian width bound\n\nOnly the Gaussian sigma upper bound changes, from 4 to 10 s; mu remains between -1 and 7 s, and sigma starts at 0.05 s. Every observed training bin enters the exact count-weighted compressed fit, while the five held-out folds, bootstrap trial draws, group exponential predictions and flat residual variance remain the saved ones. Positive contrast favours the Gaussian. Tests are two-sided Wilcoxon tests of the 15 session medians, with raw p-values.\n\n'
    text+=markdown(summary)+f'\n\nFit failures: {len(failures)}. Wall time: {seconds:.1f} s. Bound flags use absolute tolerance 1e-5. Full-data widths describe the same cells as the held-out contrasts; local nonlinear refinement is the weakest numerical step.\n'
    (OUTPUT / 'summary.md').write_text(text,encoding='utf-8')
    (OUTPUT / 'run_report.json').write_text(json.dumps(dict(seconds=seconds,failures=len(failures)),indent=2))
    print(text,flush=True)


if __name__ == '__main__':
    main()
