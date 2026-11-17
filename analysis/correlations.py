# -*- coding: utf-8 -*-
'''
Created on Mon Nov 16 13:12:22 2026

relate fitted onset to relaxation and time-cell peak to width

@author: Dinghao Luo
'''

#%% imports
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.two_exponential import signed_test
OUTPUT = ROOT / 'outputs' / 'correlations'

#%% correlations
def correlation(first_values,second_values):
    if len(first_values)<3 or np.ptp(first_values)==0 or np.ptp(second_values)==0:
        return np.nan
    return float(spearmanr(first_values,second_values).statistic)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    parameters=pd.concat([pd.read_csv(path) for path in sorted((ROOT / 'outputs' / 'wider_window_fits').glob('*_wider_parameters.csv'))],ignore_index=True)
    rows,sessions,retained=[],[],[]
    for group,first_parameter,second_parameter in [('tic_down','decay_mu','decay_tau'),('tic_up','rise_mu','rise_tau'),
                      ('time_cell','decay_mu','decay_tau'),('time_cell','gaussian_mu','gaussian_sigma')]:
        all_cells=parameters[parameters.group.eq(group)]
        cells=all_cells[~all_cells[f'{first_parameter}_at_bound'] & ~all_cells[f'{second_parameter}_at_bound']].copy()
        pooled=correlation(cells[first_parameter],cells[second_parameter])
        values=[]
        for session,part in cells.groupby('session'):
            correlation_value=correlation(part[first_parameter],part[second_parameter])
            sessions.append(dict(group=group,x=first_parameter,y=second_parameter,session=session,n=len(part),r=correlation_value))
            if np.isfinite(correlation_value):
                values.append(correlation_value)
        values=np.array(values)
        rows.append(dict(group=group,x=first_parameter,y=second_parameter,n=len(cells),excluded=len(all_cells)-len(cells),
                         pooled_r=pooled,sessions_defined=len(values),median_session_r=np.median(values),
                         session_r_q25=np.quantile(values,.25),session_r_q75=np.quantile(values,.75),wilcoxon_p=signed_test(values)))
        retained.append(cells[['session','neu_id','group',first_parameter,second_parameter]].assign(pair=f'{first_parameter}_vs_{second_parameter}'))
    summary=pd.DataFrame(rows)
    summary.to_csv(OUTPUT / 'correlation_summary.csv',index=False)
    pd.DataFrame(sessions).to_csv(OUTPUT / 'session_correlations.csv',index=False)
    pd.concat(retained,ignore_index=True).to_csv(OUTPUT / 'retained_cells.csv',index=False)


if __name__=='__main__':
    main()
