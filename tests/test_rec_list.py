# -*- coding: utf-8 -*-
'''
Created on 4 November 2026

tests for the session path helper

@author: Dinghao Luo
'''

#%% imports
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rec_list import SESSIONS_4S, session_path

#%% tests
def test_session_list_has_no_duplicates():
    assert len(set(SESSIONS_4S)) == len(SESSIONS_4S)

def test_older_session_uses_filter_subfolder():
    path = session_path('A145-20230313-02')
    assert path.as_posix().endswith(
        'immobile_2p/anmc145/A145-20230313/A145-20230313-02/test_GF_100'
        )

def test_newer_session_has_no_filter_subfolder():
    path = session_path('A214-20251013-02')
    assert path.as_posix().endswith(
        'immobile_2p/anmc214/A214-20251013/A214-20251013-02'
        )
