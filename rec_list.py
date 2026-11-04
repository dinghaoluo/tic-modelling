# -*- coding: utf-8 -*-
'''
Created on 4 November 2026

recording lists for the 4-s delay imaging sessions

@author: Dinghao Luo
'''

#%% imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'utils'))
from project_paths import IMAGING_ROOT

#%% constants
SESSIONS_4S = [
    'A132-20221124-02',
    'A132-20221124-04',
    'A133-20230120-06',
    'A133-20230114-02',
    'A143-20230315-02',
    'A145-20230313-02',
    'A147-20230401-02',
    'A148-20230331-03',
    'A163-20230903-02',
    'A164-20230902-05',
    'A213-20250916-04',
    'A214-20251009-04',
    'A215-20251010-04',
    'A214-20251013-02',
    'A215-20251013-02',
    ]

# the 2022 and 2023 sessions keep their outputs in a Gaussian-filter subfolder
_FIRST_ANIMAL_WITHOUT_SUBFOLDER = 200

#%% helpers
def session_path(session: str) -> Path:
    '''return the folder holding one session's aligned arrays'''
    animal = int(session[1:4])
    day = session.rsplit('-', 1)[0]
    path = IMAGING_ROOT / f'anmc{animal}' / day / session
    if animal < _FIRST_ANIMAL_WITHOUT_SUBFOLDER:
        path = path / 'test_GF_100'
    return path
