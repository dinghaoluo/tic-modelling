# equal time-bin weights

The full-data Gaussian, decay and rise fits give each retained model-axis coordinate equal weight, after exact compression and removal of coordinates represented in fewer than 10 trials. The rise axis is T minus t; its coordinates retain the saved sub-bin trial timing rather than being rounded to 50 ms. The default count-weighted fit is unchanged. All three models are shown for each group, including the group's non-primary exponential. Baseline a0 is unbounded, so its bound percentages are undefined.

| group | model | parameter | n | original_median | equal_median | median_absolute_difference | spearman_r | original_bound_pct | equal_bound_pct | bound_change_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tic_down | decay | a | 404 | 0.11294 | 0.11968 | 0.0066982 | 0.97497 | 0 | 0 | 0 |
| tic_down | decay | a0 | 404 | -0.002987 | -0.00096815 | 0.0039965 | 0.88867 | nan | nan | nan |
| tic_down | decay | mu | 404 | 0.34151 | 0.41944 | 0.029931 | 0.83128 | 1.9802 | 7.9208 | 5.9406 |
| tic_down | decay | tau | 404 | 7.9556 | 7.8352 | 0.053772 | 0.87044 | 48.02 | 49.01 | 0.9901 |
| tic_down | gaussian | a | 404 | 0.19239 | 0.19312 | 0.023637 | 0.89862 | 0 | 0 | 0 |
| tic_down | gaussian | a0 | 404 | -0.058086 | -0.038664 | 0.022508 | 0.84794 | nan | nan | nan |
| tic_down | gaussian | mu | 404 | 2.4554 | 2.729 | 0.098151 | 0.81061 | 1.7327 | 3.2178 | 1.4851 |
| tic_down | gaussian | sigma | 404 | 2.3427 | 2.1759 | 0.033327 | 0.81563 | 41.089 | 42.079 | 0.9901 |
| tic_down | rise | a | 377 | 0.16296 | 0.19241 | 0.063588 | 0.66233 | 0 | 0 | 0 |
| tic_down | rise | a0 | 377 | -0.0067114 | 0.006082 | 0.03577 | 0.61544 | nan | nan | nan |
| tic_down | rise | mu | 377 | 0.22679 | 0.97412 | 0.25664 | 0.62928 | 8.2228 | 11.671 | 3.4483 |
| tic_down | rise | tau | 377 | 5.5995 | 2.1065 | 1.6745 | 0.32442 | 38.992 | 33.687 | -5.305 |
| tic_up | decay | a | 479 | 0.097177 | 0.10449 | 0.0067396 | 0.9744 | 0 | 0 | 0 |
| tic_up | decay | a0 | 479 | 0.00061414 | 0.0054627 | 0.0021517 | 0.8514 | nan | nan | nan |
| tic_up | decay | mu | 479 | 1.7413 | 2.0656 | 0.060672 | 0.76287 | 13.57 | 23.8 | 10.23 |
| tic_up | decay | tau | 479 | 10 | 10 | 0 | 0.68612 | 79.958 | 72.234 | -7.7244 |
| tic_up | gaussian | a | 479 | 0.15695 | 0.16316 | 0.013443 | 0.91709 | 0 | 0 | 0 |
| tic_up | gaussian | a0 | 479 | -0.040884 | -0.033396 | 0.010757 | 0.87273 | nan | nan | nan |
| tic_up | gaussian | mu | 479 | 4.0536 | 4.2203 | 0.22706 | 0.85649 | 5.6367 | 8.5595 | 2.9228 |
| tic_up | gaussian | sigma | 479 | 2.4779 | 2.3716 | 0.18018 | 0.75376 | 29.019 | 29.228 | 0.20877 |
| tic_up | rise | a | 474 | 0.2305 | 0.26064 | 0.058365 | 0.76489 | 0 | 0 | 0 |
| tic_up | rise | a0 | 474 | -0.085092 | -0.035445 | 0.043031 | 0.58438 | nan | nan | nan |
| tic_up | rise | mu | 474 | 0.010367 | 0.074869 | 0.099237 | 0.4908 | 1.4768 | 2.9536 | 1.4768 |
| tic_up | rise | tau | 474 | 5.703 | 2.9598 | 1.1904 | 0.41772 | 40.506 | 35.654 | -4.8523 |
| time_cell | decay | a | 1177 | 0.39602 | 0.4243 | 0.02512 | 0.95824 | 0 | 0 | 0 |
| time_cell | decay | a0 | 1177 | -0.018499 | -0.037585 | 0.022751 | 0.79451 | nan | nan | nan |
| time_cell | decay | mu | 1177 | -0.45936 | -0.66966 | 0.021736 | 0.88811 | 33.05 | 32.625 | -0.42481 |
| time_cell | decay | tau | 1177 | 1.6604 | 1.7939 | 0.16992 | 0.75429 | 6.3721 | 9.6856 | 3.3135 |
| time_cell | gaussian | a | 1177 | 0.25972 | 0.26661 | 0.010487 | 0.9277 | 0 | 0 | 0 |
| time_cell | gaussian | a0 | 1177 | -0.024462 | -0.029599 | 0.012366 | 0.90942 | nan | nan | nan |
| time_cell | gaussian | mu | 1177 | 0.46181 | 0.52289 | 0.0082485 | 0.92054 | 6.4571 | 7.7315 | 1.2744 |
| time_cell | gaussian | sigma | 1177 | 1.0596 | 1.0938 | 0.070949 | 0.85146 | 2.0391 | 3.9932 | 1.9541 |
| time_cell | rise | a | 1134 | 0.19298 | 0.25489 | 0.050238 | 0.72812 | 0 | 0 | 0 |
| time_cell | rise | a0 | 1134 | 0.011716 | -0.0080404 | 0.02054 | 0.70521 | nan | nan | nan |
| time_cell | rise | mu | 1134 | 2.6639 | 2.5311 | 0.17315 | 0.64218 | 36.067 | 31.217 | -4.8501 |
| time_cell | rise | tau | 1134 | 10 | 10 | 0.079987 | 0.4892 | 61.023 | 63.58 | 2.5573 |

Unsupported fits: 75; attempted fits that did not converge: 0. Wall time: 110.9 s. Unsupported fits are listed in failures.csv, and parameter summaries use the defined matched subset. These paired descriptive checks combine equal weighting with the requested 10-trial support restriction.

| session | model | issue | n |
| --- | --- | --- | --- |
| A215-20251013-02 | rise | fewer than four coordinates with 10 trials | 75 |

A215-20251013-02 has 125 saved trials but only two exact rise-axis coordinates with at least 10 contributing trials, leaving its four-parameter rise fit unidentified; the 75 unsupported fits comprise 27 ramp-down TICs, five ramp-up TICs and 43 time cells. No rounding or rebinning was introduced to obtain additional coordinates.
