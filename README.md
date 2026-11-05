# tic-modelling

This repository contains my code for fitting parametric receptive fields to dorsal CA1 activity in the immobile time-estimation task of Citrin, Heldman, Ye & Wang ([bioRxiv, 2026](https://doi.org/10.64898/2026.04.12.716593)). Between the last lick of one trial and the first lick of the next, each neuron's dF/F on every trial is fitted with a Gaussian field and with exponential shapes (a decay from the last lick, a rise towards the first lick), alongside a linear ramp and a plateau as guards; each shape is allowed to rescale with that trial's interval and is passed through the calcium kernel before it meets the data, and the shapes are compared by held-out log-likelihood. The fitted parameters are then compared between functionally defined populations and against published time cells ([Cao et al., 2022](https://elifesciences.org/articles/75353)) and ramping cells ([Cao, Bright & Howard, 2024](https://sites.bu.edu/tcn/files/2024/09/Cao_et_al_2024.pdf)).

The recordings remain on the network drive of Max Planck Florida Institute for Neuroscience, and the paths that point to them are collected in `utils/project_paths.py`; nothing in this repository writes to those folders. Generated data, figures, outputs, and working notes are stored under `data/`, `figures/`, `outputs/`, and `notes/`, all of which are ignored by Git so that reproduction can be performed locally.

## repository layout

- `utils/`: shared paths and helpers
- `tests/`: tests for the shared code
- `rec_list.py`: the recording list of the 4-s delay sessions
- `PROGRESS.md`: dated record of what has and has not been run

## setup

The dedicated `tic-modelling` environment is a Python 3.11 virtual environment at `Z:\Dinghao\code\tic-modelling\data\tic-modelling`, created from the existing user-profile Python because Miniforge was not available. Its interpreter is `Z:\Dinghao\code\tic-modelling\data\tic-modelling\Scripts\python.exe`; the system Python is unchanged, and the environment stays in the ignored data directory.

From the repository root in PowerShell:

```powershell
& .\data\tic-modelling\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -B -m pytest -p no:cacheprovider tests
```

Without activation, call `& .\data\tic-modelling\Scripts\python.exe` in place of `python` for every analysis command.
