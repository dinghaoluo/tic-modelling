'''Created on 4 November 2026
reproduce the specified lick ratios without replacing source-table labels
@author: Dinghao Luo
'''

#%% imports
import json
import sys
import time
from pathlib import Path

import matio
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rec_list import SESSIONS_4S, session_path
from utils.matlab_inputs import Fluorescence, load_numeric
from utils.project_paths import TABLES_ROOT

#%% main
def main():
    started = time.perf_counter()
    output = ROOT / 'outputs' / 'fits_v1'
    labels = pd.read_csv(ROOT / 'data' / 'labels.csv')
    tic_source = matio.load_from_mat(str(TABLES_ROOT / 'RiseLastLickDownLickNeurons_thres.mat'))
    ids = tic_source['RiseLLDownLickNeuronID']
    expected = set(zip(ids.rec_name, ids.neu_id.astype(int)))
    rows, checks = [], []
    for session in SESSIONS_4S:
        folder = session_path(session)
        selected = load_numeric(folder / 'FirstLick_LL.mat')['Ind_35to45'].ravel()[1:].astype(bool)
        ratios = {}
        for alignment, filename in [('last', 'dFFGF_befLastLick.mat'), ('first', 'dFFGF_befLick.mat')]:
            with Fluorescence(folder / filename) as reader:
                if len(selected) != reader.shape(0)[0]:
                    raise ValueError(f'{session}: shifted Ind_35to45 does not match trial rows')
                checks.append(dict(session=session, alignment=alignment, format=reader.format,
                                   neurons=reader.n_neurons, trial_rows=reader.shape(0)[0]))
                for neuron in range(reader.n_neurons):
                    arr = reader.read(neuron)
                    if arr.shape[1] != 5000:
                        raise ValueError(f'{session}: unexpected sample count')
                    for subset, mask in [('all', np.ones(len(selected), dtype=bool)), ('Ind_35to45', selected)]:
                        profile = np.mean(arr[mask], axis=0)
                        profile = (profile - profile.min()) / (profile.max() - profile.min())
                        ratio = np.mean(profile[1500:1750]) / np.mean(profile[1250:1500])
                        ratios[(neuron + 1, subset, alignment)] = float(ratio)
        for neuron in range(1, reader.n_neurons + 1):
            for subset in ['all', 'Ind_35to45']:
                last = ratios[(neuron, subset, 'last')]
                first = ratios[(neuron, subset, 'first')]
                rows.append(dict(session=session, neu_id=neuron, subset=subset,
                                 ratio_last=last, ratio_first=first,
                                 reproduced_tic=last > 1.5 and first < 2 / 3,
                                 source_tic=(session, neuron) in expected))
        pd.DataFrame(rows).to_csv(output / 'reproduced_ratios.csv', index=False)
        pd.DataFrame(checks).to_csv(output / 'file_formats.csv', index=False)
        print(session, 'ratio reproduction complete', flush=True)
    frame = pd.DataFrame(rows)
    summary = {}
    for subset in ['all', 'Ind_35to45']:
        selected_rows = frame[(frame.subset == subset) & frame.reproduced_tic]
        found = set(zip(selected_rows.session, selected_rows.neu_id))
        summary[subset] = dict(source_tics=len(expected), recovered=len(found & expected),
                               extras=len(found - expected), missing=len(expected - found))
    summary['time_cells'] = int(labels.is_time_cell.sum())
    summary['tic_and_time_cell'] = int((labels.is_tic & labels.is_time_cell).sum())
    summary['seconds'] = time.perf_counter() - started
    (output / 'reproduction.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
