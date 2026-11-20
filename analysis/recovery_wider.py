'''
Created on Fri Nov 20 2026

repeat the sampled-cell recovery check on the wider window

@author: Dinghao Luo
'''

#%% imports
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from shutil import copyfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rec_list import SESSIONS_4S
from analysis.recovery_check import OUTPUT, WIDER_RECOVERY, recovery_session, recovery_summary


#%% run
def main():
    WIDER_RECOVERY.mkdir(parents=True, exist_ok=True)
    for name in ['recovery_sample.csv', 'recovery_allocation.csv']:
        copyfile(OUTPUT / name, WIDER_RECOVERY / name)
    workers = min(12, os.cpu_count() or 1, len(SESSIONS_4S))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(partial(recovery_session, wider=True), SESSIONS_4S))
    recovery_summary(WIDER_RECOVERY)


if __name__ == '__main__':
    main()
