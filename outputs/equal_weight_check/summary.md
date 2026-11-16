# equal time-bin weights

The full-data Gaussian and decay fits give each retained exact model-axis coordinate equal weight, after removal of coordinates represented in fewer than 10 trials. The rise refits instead average observations within 50 ms bins of T minus t, give each bin equal weight, and retain bins supported by at least 10 distinct trials. Bins are left-closed, right-open, with edges at integer multiples of 50 ms; predictions are evaluated at bin centres. The default count-weighted fit is unchanged. All three models are shown for each group, including the group's non-primary exponential. Baseline a0 is unbounded, so its bound percentages are undefined.

## retained rise bins per session

| session | cells | minimum | median | maximum |
| --- | --- | --- | --- | --- |
| A132-20221124-02 | 253 | 143 | 143 | 143 |
| A132-20221124-04 | 243 | 149 | 149 | 149 |
| A133-20230114-02 | 172 | 127 | 127 | 127 |
| A133-20230120-06 | 79 | 146 | 146 | 146 |
| A143-20230315-02 | 101 | 151 | 151 | 151 |
| A145-20230313-02 | 85 | 137 | 137 | 137 |
| A147-20230401-02 | 135 | 128 | 128 | 128 |
| A148-20230331-03 | 101 | 134 | 134 | 134 |
| A163-20230903-02 | 112 | 125 | 125 | 125 |
| A164-20230902-05 | 130 | 112 | 112 | 112 |
| A213-20250916-04 | 105 | 137 | 137 | 137 |
| A214-20251009-04 | 120 | 137 | 137 | 137 |
| A214-20251013-02 | 162 | 122 | 122 | 122 |
| A215-20251010-04 | 187 | 134 | 134 | 134 |
| A215-20251013-02 | 75 | 128 | 128 | 128 |

## parameter comparisons

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
| tic_down | rise | a | 404 | 0.15803 | 0.18562 | 0.030128 | 0.91186 | 0 | 0 | 0 |
| tic_down | rise | a0 | 404 | -0.0087381 | -0.044047 | 0.022819 | 0.82085 | nan | nan | nan |
| tic_down | rise | mu | 404 | 0.22332 | 0.13475 | 0.04024 | 0.84695 | 7.9208 | 7.6733 | -0.24752 |
| tic_down | rise | tau | 404 | 5.1349 | 7.6491 | 0.20322 | 0.71316 | 38.366 | 44.554 | 6.1881 |
| tic_up | decay | a | 479 | 0.097177 | 0.10449 | 0.0067396 | 0.9744 | 0 | 0 | 0 |
| tic_up | decay | a0 | 479 | 0.00061414 | 0.0054627 | 0.0021517 | 0.8514 | nan | nan | nan |
| tic_up | decay | mu | 479 | 1.7413 | 2.0656 | 0.060672 | 0.76287 | 13.57 | 23.8 | 10.23 |
| tic_up | decay | tau | 479 | 10 | 10 | 0 | 0.68612 | 79.958 | 72.234 | -7.7244 |
| tic_up | gaussian | a | 479 | 0.15695 | 0.16316 | 0.013443 | 0.91709 | 0 | 0 | 0 |
| tic_up | gaussian | a0 | 479 | -0.040884 | -0.033396 | 0.010757 | 0.87273 | nan | nan | nan |
| tic_up | gaussian | mu | 479 | 4.0536 | 4.2203 | 0.22706 | 0.85649 | 5.6367 | 8.5595 | 2.9228 |
| tic_up | gaussian | sigma | 479 | 2.4779 | 2.3716 | 0.18018 | 0.75376 | 29.019 | 29.228 | 0.20877 |
| tic_up | rise | a | 479 | 0.2305 | 0.23562 | 0.022253 | 0.95345 | 0 | 0 | 0 |
| tic_up | rise | a0 | 479 | -0.085026 | -0.091108 | 0.021744 | 0.84058 | nan | nan | nan |
| tic_up | rise | mu | 479 | 0.011467 | -0.0035075 | 0.029929 | 0.84064 | 1.4614 | 2.0877 | 0.6263 |
| tic_up | rise | tau | 479 | 5.6394 | 6.2049 | 0.20064 | 0.79177 | 40.292 | 43.006 | 2.714 |
| time_cell | decay | a | 1177 | 0.39602 | 0.4243 | 0.02512 | 0.95824 | 0 | 0 | 0 |
| time_cell | decay | a0 | 1177 | -0.018499 | -0.037585 | 0.022751 | 0.79451 | nan | nan | nan |
| time_cell | decay | mu | 1177 | -0.45936 | -0.66966 | 0.021736 | 0.88811 | 33.05 | 32.625 | -0.42481 |
| time_cell | decay | tau | 1177 | 1.6604 | 1.7939 | 0.16992 | 0.75429 | 6.3721 | 9.6856 | 3.3135 |
| time_cell | gaussian | a | 1177 | 0.25972 | 0.26661 | 0.010487 | 0.9277 | 0 | 0 | 0 |
| time_cell | gaussian | a0 | 1177 | -0.024462 | -0.029599 | 0.012366 | 0.90942 | nan | nan | nan |
| time_cell | gaussian | mu | 1177 | 0.46181 | 0.52289 | 0.0082485 | 0.92054 | 6.4571 | 7.7315 | 1.2744 |
| time_cell | gaussian | sigma | 1177 | 1.0596 | 1.0938 | 0.070949 | 0.85146 | 2.0391 | 3.9932 | 1.9541 |
| time_cell | rise | a | 1177 | 0.18757 | 0.20609 | 0.020604 | 0.95745 | 0 | 0 | 0 |
| time_cell | rise | a0 | 1177 | 0.012165 | 0.013121 | 0.0046925 | 0.95871 | nan | nan | nan |
| time_cell | rise | mu | 1177 | 2.6465 | 2.8513 | 0.021808 | 0.88508 | 35.769 | 44.69 | 8.921 |
| time_cell | rise | tau | 1177 | 10 | 10 | 1e-09 | 0.76162 | 60.833 | 56.754 | -4.0782 |

Unsupported fits: 0; attempted fits that did not converge: 0. Wall time for rise refits only: 55.1 s. Unsupported fits are listed in failures.csv, and parameter summaries use the defined matched subset. These paired descriptive checks combine equal weighting with the requested 10-trial support restriction.
