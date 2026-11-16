'''Created on 16 November 2026
relate fitted onset to relaxation and time-cell peak to width
@author: Dinghao Luo
'''

#%% imports
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.two_exponential import markdown, signed_test
OUTPUT = ROOT / 'outputs' / 'correlations'

#%% correlations
def correlation(x,y):
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:
        return np.nan
    return float(spearmanr(x,y).statistic)


def main():
    started=time.perf_counter()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    parameters=pd.concat([pd.read_csv(p) for p in sorted((ROOT / 'outputs' / 'undecided_check').glob('*_wider_parameters.csv'))],ignore_index=True)
    rows,sessions,retained=[],[],[]
    for group,x,y in [('tic_down','decay_mu','decay_tau'),('tic_up','rise_mu','rise_tau'),
                      ('time_cell','decay_mu','decay_tau'),('time_cell','gaussian_mu','gaussian_sigma')]:
        all_cells=parameters[parameters.group.eq(group)]
        cells=all_cells[~all_cells[f'{x}_at_bound'] & ~all_cells[f'{y}_at_bound']].copy()
        pooled=correlation(cells[x],cells[y])
        values=[]
        for session,part in cells.groupby('session'):
            r=correlation(part[x],part[y])
            sessions.append(dict(group=group,x=x,y=y,session=session,n=len(part),r=r))
            if np.isfinite(r):
                values.append(r)
        values=np.array(values)
        rows.append(dict(group=group,x=x,y=y,n=len(cells),excluded=len(all_cells)-len(cells),
                         pooled_r=pooled,sessions_defined=len(values),median_session_r=np.median(values),
                         session_r_q25=np.quantile(values,.25),session_r_q75=np.quantile(values,.75),wilcoxon_p=signed_test(values)))
        retained.append(cells[['session','neu_id','group',x,y]].assign(pair=f'{x}_vs_{y}'))
    summary=pd.DataFrame(rows)
    summary.to_csv(OUTPUT / 'correlation_summary.csv',index=False)
    pd.DataFrame(sessions).to_csv(OUTPUT / 'session_correlations.csv',index=False)
    pd.concat(retained,ignore_index=True).to_csv(OUTPUT / 'retained_cells.csv',index=False)
    seconds=time.perf_counter()-started
    text='# latency, relaxation and peak width\n\nBoth parameters must be away from either saved bound; exclusion uses the wider-window bound flags, with no refitting. Spearman correlations are computed across retained cells and separately within each session; raw two-sided Wilcoxon p-values test the defined session correlations against zero. Rise mu retains its saved time-before-next-lick definition. IQRs describe variation between sessions, not uncertainty in a pooled coefficient.\n\n'
    text+=markdown(summary)+f'\n\nWall time: {seconds:.1f} s. Excluding bounded cells can alter the sampled relationship; the pooled and session estimates are reported together so a heavily represented session can be recognised.\n'
    (OUTPUT / 'summary.md').write_text(text,encoding='utf-8')
    (OUTPUT / 'run_report.json').write_text(json.dumps(dict(seconds=seconds), indent=2))
    print(text,flush=True)


if __name__=='__main__':
    main()
