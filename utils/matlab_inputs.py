'''Created on 4 November 2026
read aligned fluorescence arrays without changing their MATLAB sources
@author: Dinghao Luo
'''

#%% imports
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

#%% readers
class Fluorescence:
    '''keep v7.3 cells on disk and restore MATLAB trials-by-samples orientation'''

    def __init__(self, path: Path):
        self.path = Path(path)
        self.format = 'v7.3' if h5py.is_hdf5(path) else 'v5'
        self.file = h5py.File(path, 'r') if self.format == 'v7.3' else None
        if self.file is not None:
            self.cells = np.asarray(self.file['dFFGFArray']).ravel(order='F')
        else:
            self.cells = loadmat(str(path))['dFFGFArray'].ravel(order='F')
        self.n_neurons = len(self.cells)

    def shape(self, neuron: int):
        if self.file is not None:
            return self.file[self.cells[neuron]].shape[::-1]
        return self.cells[neuron].shape

    def read(self, neuron: int, start=0, end=5000):
        if self.file is not None:
            return np.asarray(self.file[self.cells[neuron]][start:end, :]).T
        return np.asarray(self.cells[neuron][:, start:end])

    def close(self):
        if self.file is not None:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def load_numeric(path: Path):
    if h5py.is_hdf5(path):
        with h5py.File(path, 'r') as f:
            return {key: np.asarray(value).T for key, value in f.items()
                    if not key.startswith('#')}
    return {key: value for key, value in loadmat(str(path)).items()
            if not key.startswith('__')}

