# fit quality of trial-averaged activity

The saved full-data parameters are evaluated on the original trial bins, and predictions inherit each cell’s observed NaN mask before averaging; no parameters are refitted. The exponential is a decay for ramp-down TICs and time cells, and a rise for ramp-up TICs.

Last-lick averages use the saved t bins, including the pre-last-lick second. Next-lick averages use time before the next lick (T_k minus t) on a 0.05 s grid starting at zero; observed values and both predictions are linearly interpolated within each trial’s shared observed support, without extrapolation or interpolation across missing endpoints. Every retained bin has at least 10 contributing trials.

R² is 1 minus the squared error of the averaged prediction divided by the squared deviation of the averaged observed trace from its mean, with equal weight for each retained time bin; negative values remain negative, while constant traces or fewer than two shared bins give an undefined value. Ramp-up TICs use next-lick alignment as primary. Differences below are Gaussian minus exponential, computed within each cell before taking the median.

| Group | Alignment | n (defined) | Gaussian median [IQR] | Exponential median [IQR] | Median difference | Gaussian >0.5 (%) | Exponential >0.5 (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| Ramp-down TIC | last_lick | 404 (404) | 0.645 [0.451, 0.796] | 0.614 [0.434, 0.752] | 0.024 | 68.8 | 66.1 |
| Ramp-up TIC | next_lick | 479 (479) | 0.725 [0.538, 0.848] | 0.793 [0.625, 0.898] | -0.051 | 77.7 | 85.0 |
| Ramp-up TIC | last_lick | 479 (479) | 0.782 [0.646, 0.882] | 0.592 [0.418, 0.739] | 0.132 | 87.1 | 63.9 |
| Time cells | last_lick | 1177 (1177) | 0.877 [0.684, 0.949] | 0.726 [0.549, 0.848] | 0.079 | 87.2 | 79.8 |

The IQR describes variation between cells rather than uncertainty in the population median. These are descriptive fits to the same trials used for fitting, so agreement with the mean can be optimistic; the saved held-out trial comparison is the check that separates reproducible shape from fitting the noise.

## example cells

Within each group’s primary alignment, cells are ordered by the larger of their two R² values, with session and neuron ID breaking ties. The selected zero-based rank is the nearest integer to (n minus 1) times the requested percentile; columns run from the 20th to the 80th percentile. No cells are selected by visual appearance.

| Group | Percentile | Session | neu_id | Gaussian R² | Exponential R² | Retained bins |
|---|---:|---|---:|---:|---:|---:|
| tic_down | 20 | A215-20251010-04 | 269 | 0.531 | 0.533 | 133 |
| tic_down | 40 | A163-20230903-02 | 103 | 0.249 | 0.674 | 125 |
| tic_down | 60 | A147-20230401-02 | 204 | 0.784 | 0.613 | 127 |
| tic_down | 80 | A133-20230114-02 | 88 | 0.850 | 0.851 | 126 |
| tic_up | 20 | A164-20230902-05 | 411 | 0.437 | 0.608 | 111 |
| tic_up | 40 | A133-20230120-06 | 11 | 0.715 | 0.769 | 145 |
| tic_up | 60 | A213-20250916-04 | 167 | 0.863 | 0.839 | 136 |
| tic_up | 80 | A163-20230903-02 | 165 | 0.840 | 0.924 | 124 |
| time_cell | 20 | A214-20251009-04 | 277 | 0.592 | 0.638 | 136 |
| time_cell | 40 | A164-20230902-05 | 74 | 0.832 | 0.749 | 112 |
| time_cell | 60 | A132-20221124-02 | 112 | 0.916 | 0.838 | 143 |
| time_cell | 80 | A143-20230315-02 | 144 | 0.960 | 0.901 | 150 |

## input observations

The three saved fit_success=False flags correspond to cross-validation fold failures in the saved session status files; none records a failure of the full-data fit used here.

The 15 saved sessions contain 404 ramp-down TICs, 479 ramp-up TICs and 1,177 time cells. The saved t bins are 0.05 s apart, running from -0.976 to 6.974 s; neuron IDs are matched explicitly rather than treated as array indices. All retained bins and R² values use the original dF/F units.

- A133-20230114-02, neuron 377 (tic_down): saved fit_success=False; retained
- A213-20250916-04, neuron 102 (tic_up): saved fit_success=False; retained
- A213-20250916-04, neuron 130 (time_cell): saved fit_success=False; retained
