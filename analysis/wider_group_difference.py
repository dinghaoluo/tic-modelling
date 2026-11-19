'''
Created on Wed Nov 18 2026

compare the saved last-lick Gaussian minus decay contrast across cell groups

@author: Dinghao Luo
'''

#%% imports
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rec_list import SESSIONS_4S
from analysis.compare_models import paired_test
from analysis.recovery_check import GROUPS, OUTPUT as WIDER

OUTPUT = ROOT / 'outputs' / 'wider_group_difference'
PAIRS = [('tic_down', 'time_cell'), ('tic_up', 'time_cell'), ('tic_down', 'tic_up')]


#%% group differences
def main():
    saved = pd.concat([pd.read_csv(WIDER / f'{session}_wider_contrasts.csv')
                       for session in SESSIONS_4S], ignore_index=True)
    cells = saved[saved.contrast.eq('gaussian_minus_decay') & saved.group.isin(GROUPS)]
    medians = cells.pivot_table(index='session', columns='group', values='mean_contrast',
                               aggfunc='median').reindex(index=SESSIONS_4S, columns=GROUPS)
    if cells.duplicated(['session', 'neu_id']).any() or not np.isfinite(cells.mean_contrast).all():
        raise ValueError('duplicate cells or non-finite saved contrasts')
    if medians.isna().any().any():
        raise ValueError('missing group median in at least one saved session')

    rows = []
    for left, right in PAIRS:
        differences = medians[left] - medians[right]
        wilcoxon_stat, wilcoxon_p = paired_test(differences)
        a = cells.loc[cells.group.eq(left), 'mean_contrast']
        b = cells.loc[cells.group.eq(right), 'mean_contrast']
        mann_whitney = mannwhitneyu(a, b, alternative='two-sided')
        rows.append(dict(group_a=left, group_b=right, n_sessions=len(differences),
                         n_cells_a=len(a), n_cells_b=len(b),
                         median_paired_difference=differences.median(),
                         q25_paired_difference=differences.quantile(.25),
                         q75_paired_difference=differences.quantile(.75),
                         wilcoxon_statistic=wilcoxon_stat, wilcoxon_p=wilcoxon_p,
                         mann_whitney_u=mann_whitney.statistic, mann_whitney_p=mann_whitney.pvalue))
    tests = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    medians.to_csv(OUTPUT / 'session_medians.csv')
    tests.to_csv(OUTPUT / 'group_tests.csv', index=False)
    print(medians.to_string())
    print(tests.to_string(index=False))


if __name__ == '__main__':
    main()
