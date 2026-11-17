# tic-modelling

This repository contains my code for fitting parametric receptive fields to dorsal CA1 activity in the immobile time-estimation task of Citrin, Heldman, Ye & Wang ([bioRxiv, 2026](https://doi.org/10.64898/2026.04.12.716593)). Between the last lick of one trial and the first lick of the next, each neuron's dF/F is fitted on the observed bins of individual trials with a Gaussian field and with exponential shapes (a decay from the last lick and a rise towards the next lick), with flat and polynomial comparisons. The exponential shapes use the exponentially modified Gaussian of Cao et al. (2024), with a fixed 0.2 s onset width; the fits do not pass through a calcium kernel. Models are compared by held-out trial errors, expressed as contrasts using the same flat-residual variance for each cell, while the two-exponential analyses compare a two-exponential model anchored to either lick. The fitted parameters are then compared between functionally defined populations and against published time cells ([Cao et al., 2022](https://elifesciences.org/articles/75353)) and ramping cells ([Cao, Bright & Howard, 2024](https://sites.bu.edu/tcn/files/2024/09/Cao_et_al_2024.pdf)).

The recordings remain on the network drive of Max Planck Florida Institute for Neuroscience, and the paths that point to them are collected in `utils/project_paths.py`; nothing in this repository writes to those folders. Generated data, figures, outputs, and working notes are stored under `data/`, `figures/`, `outputs/`, and `notes/`, with selected result tables and figures tracked alongside the code.

## repository layout

- `analysis/`: dataset construction, label reproduction, held-out fits and comparisons, parameter summaries, and the final figures
- `utils/`: receptive-field models, trial and mean-trace scores, MATLAB readers, and paths
- `tests/`: tests for the retained analyses and shared code
- `data/binned/ and data/binned_wider/`: trial-binned fluorescence for the interval and wider windows
- `data/valid_trials/`: the adjacent-trial validity rule
- `outputs/dataset/, outputs/fits/ and outputs/wider_window_fits/`: source labels and alignment tables, interval fits, and wider-window fits
- `outputs/two_exponential/ and outputs/revised_two_exponential/`: bump parameters, held-out contrasts and averaged-trace scores
- `outputs/fit_quality/, outputs/fit_quality_two_exponential/, outputs/equal_weight_check/, outputs/gaussian_wide/, outputs/trial_threshold/ and outputs/correlations/`: fitted-trace and parameter comparisons
- `outputs/model_comparison/, outputs/parameter_comparison/, outputs/peak_position/ and outputs/label_stability/`: saved tables used by the retained analyses
- `figures/`: manuscript figures, written directly into this folder
- `_archive/`: superseded analyses, tests and plotting scripts under their former paths
- `rec_list.py`: the recording list of the 4-s delay sessions
- `PROGRESS.md`: dated record of what has and has not been run

## setup

Interpreter: `Z:\Dinghao\code\tic-modelling\data\tic-modelling\Scripts\python.exe`

```powershell
& .\data\tic-modelling\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -B -m pytest -p no:cacheprovider tests
```
