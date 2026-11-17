# -*- coding: utf-8 -*-
'''
Created on Wed Nov  4 16:26:12 2026

project path stems for the TIC receptive-field repository

@author: Dinghao Luo
'''

#%% imports
from pathlib import Path

#%% locate the repository root
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent

#%% in-repo generated-output roots
REPO_ROOT    = _REPO_ROOT
DATA_ROOT    = _REPO_ROOT / 'data'
FIGURES_ROOT = _REPO_ROOT / 'figures'
OUTPUT_ROOT  = _REPO_ROOT / 'outputs'
NOTES_ROOT   = _REPO_ROOT / 'notes'

#%% raw-input roots (read only)
KORI_ROOT        = Path(r'Z:/Kori')
IMAGING_ROOT     = KORI_ROOT / 'immobile_2p'
CODE_OUTPUT_ROOT = KORI_ROOT / 'immobile_code'
TABLES_ROOT      = (
    CODE_OUTPUT_ROOT / 'RiseDown' / 'tables' / '2025-05to05' / 'Variable' / 'Path4sData'
    )
EPHYS_TABLES_ROOT = CODE_OUTPUT_ROOT / 'Ephys' / 'Oct2026'
