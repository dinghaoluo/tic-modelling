# -*- coding: utf-8 -*-
'''
Created on Mon Nov 16 13:12:22 2026

apply the lick thresholds to single trials and paired trial removals

@author: Dinghao Luo
'''

#%% imports
import os
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rec_list import session_path, SESSIONS_4S
from utils.matlab_inputs import Fluorescence

OUTPUT  = ROOT / 'outputs' / 'trial_threshold'
GROUPS  = ['tic_down','tic_up','time_cell','other','tic_and_time_cell']
REPEATS = 200
SEED    = 20261116

#%% threshold helpers
def profile_pass(profiles, last):
    profiles=np.atleast_2d(profiles)
    minimum=profiles.min(axis=1)
    span=profiles.max(axis=1)-minimum
    pre=profiles[:,1250:1500].mean(axis=1)-minimum
    post=profiles[:,1500:1750].mean(axis=1)-minimum
    defined=np.isfinite(profiles).all(axis=1) & (span>0)
    # the positive normalisation range cancels in the threshold difference
    value=post-(1.5 if last else 2/3)*pre
    return defined & ((value>0) if last else (value<0)),defined


def trial_pass(array,last):
    profile=array.mean(axis=0)
    minimum=profile.min()
    span=np.ptp(profile)
    if not np.isfinite(span) or span<=0:
        return np.full(len(array),np.nan)
    pre=(array[:,1250:1500].mean(axis=1)-minimum)/span
    post=(array[:,1500:1750].mean(axis=1)-minimum)/span
    return (post-1.5*pre>0) if last else (post-(2/3)*pre<0)


def removed_trials(array,fraction):
    n_drop=int(np.ceil(fraction*len(array)))
    score=array[:,1500:1750].mean(axis=1)
    return np.argsort(-score,kind='stable')[:n_drop]


def session_job(session):
    labels=pd.read_csv(ROOT / 'outputs' / 'dataset' / 'labels.csv')
    labels=labels[labels.session.eq(session)].sort_values('neu_id')
    folder=session_path(session)
    cells,drops,null,issues=[],[],[],[]
    rng=np.random.default_rng(np.random.SeedSequence([SEED,SESSIONS_4S.index(session)]))
    with Fluorescence(folder / 'dFFGF_befLastLick.mat') as last_reader, Fluorescence(folder / 'dFFGF_befLick.mat') as next_reader:
        np.testing.assert_array_equal(labels.neu_id,np.arange(1,last_reader.n_neurons+1))
        assert last_reader.n_neurons==next_reader.n_neurons
        for cell in labels.itertuples():
            last=last_reader.read(int(cell.neu_id)-1).astype(float)
            next_array=next_reader.read(int(cell.neu_id)-1).astype(float)
            if last.shape!=next_array.shape or last.shape[1]!=5000:
                raise ValueError(f'{session}/{cell.neu_id}: trial arrays do not match')
            if not np.isfinite(last).all() or not np.isfinite(next_array).all():
                raise ValueError(f'{session}/{cell.neu_id}: non-finite raw trial samples')
            pass_last=trial_pass(last,True)
            pass_next=trial_pass(next_array,False)
            full_last,defined_last=profile_pass(last.mean(axis=0),True)
            full_next,defined_next=profile_pass(next_array.mean(axis=0),False)
            defined=bool(defined_last[0] and defined_next[0])
            full_pass=bool(full_last[0] and full_next[0])
            if not defined:
                issues.append(dict(session=session,neu_id=cell.neu_id,issue='constant averaged trace'))
            if full_pass!=bool(cell.is_tic):
                issues.append(dict(session=session,neu_id=cell.neu_id,issue='all-trial threshold differs from saved TIC label'))
            fractions=[100*np.mean(pass_last),100*np.mean(pass_next),
                       100*np.mean(np.asarray(pass_last,dtype=float)*np.asarray(pass_next,dtype=float))]
            cells.append(dict(session=session,neu_id=cell.neu_id,group=cell.group,is_tic=cell.is_tic,
                              n_trials=len(last),pct_last=fractions[0],pct_next=fractions[1],pct_both=fractions[2],
                              full_pass=full_pass,defined=defined))
            if cell.is_tic:
                last_sum=last.sum(axis=0)
                next_sum=next_array.sum(axis=0)
                for fraction in [.1,.2,.3]:
                    removed=removed_trials(last,fraction)
                    kept=len(last)-len(removed)
                    last_mean=(last_sum-last[removed].sum(axis=0))/kept
                    next_mean=(next_sum-next_array[removed].sum(axis=0))/kept
                    lp,ld=profile_pass(last_mean,True)
                    npass,nd=profile_pass(next_mean,False)
                    drops.append(dict(session=session,neu_id=cell.neu_id,group=cell.group,drop_pct=int(100*fraction),
                                      n_drop=len(removed),passing=bool(lp[0] and npass[0]),defined=bool(ld[0] and nd[0])))
                    removal=np.zeros((REPEATS,len(last)))
                    for repeat in range(REPEATS):
                        removal[repeat,rng.choice(len(last),size=len(removed),replace=False)]=1.
                    last_mean=(last_sum-removal@last)/kept
                    next_mean=(next_sum-removal@next_array)/kept
                    lp,ld=profile_pass(last_mean,True)
                    npass,nd=profile_pass(next_mean,False)
                    for repeat in range(REPEATS):
                        null.append(dict(session=session,neu_id=cell.neu_id,group=cell.group,drop_pct=int(100*fraction),
                                         repeat=repeat,passing=bool(lp[repeat] and npass[repeat]),defined=bool(ld[repeat] and nd[repeat])))
    pd.DataFrame(cells).to_csv(OUTPUT / f'{session}_cells.csv',index=False)
    pd.DataFrame(drops).to_csv(OUTPUT / f'{session}_drops.csv',index=False)
    pd.DataFrame(null).to_csv(OUTPUT / f'{session}_null.csv',index=False)
    status=dict(session=session, issues=issues)
    pd.DataFrame(issues, columns=['session', 'neu_id', 'issue']).to_csv(OUTPUT / f'{session}_issues.csv', index=False)
    return status


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        statuses=list(pool.map(session_job,SESSIONS_4S))
    frames={suffix:pd.concat([pd.read_csv(OUTPUT / f'{session}_{suffix}.csv') for session in SESSIONS_4S],ignore_index=True)
            for suffix in ['cells','drops','null']}
    for suffix,frame in frames.items():
        frame.to_csv(OUTPUT / f'{suffix}.csv',index=False)
    cells,drops,null=[frames[key] for key in ['cells','drops','null']]
    rows=[]
    for group in GROUPS+['all_TICs']:
        part=cells[cells.is_tic] if group=='all_TICs' else cells[cells.group.eq(group)]
        for threshold in ['last','next','both']:
            values=part[f'pct_{threshold}']
            rows.append(dict(group=group,threshold=threshold,n=len(part),defined=int(values.notna().sum()),
                             median=values.median(),q25=values.quantile(.25),q75=values.quantile(.75),
                             pct_cells_above_50=100*(values>50).mean()))
    single=pd.DataFrame(rows)
    single.to_csv(OUTPUT / 'trial_summary.csv',index=False)
    rows=[]
    for group in ['tic_down','tic_up','tic_and_time_cell','all_TICs']:
        part=drops if group=='all_TICs' else drops[drops.group.eq(group)]
        random=null if group=='all_TICs' else null[null.group.eq(group)]
        for fraction,chunk in part.groupby('drop_pct'):
            draws=100*random[random.drop_pct.eq(fraction)].groupby('repeat').passing.mean()
            rows.append(dict(group=group,drop_pct=fraction,n=len(chunk),top_retained_pct=100*chunk.passing.mean(),
                             random_mean_pct=draws.mean(),random_q025=draws.quantile(.025),random_q975=draws.quantile(.975),repeats=len(draws)))
    dropping=pd.DataFrame(rows)
    dropping.to_csv(OUTPUT / 'drop_summary.csv',index=False)
    issues=[issue for status in statuses for issue in status['issues']]
    pd.DataFrame(issues,columns=['session','neu_id','issue']).to_csv(OUTPUT / 'issues.csv',index=False)


if __name__=='__main__':
    main()
