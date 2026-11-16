# Gaussian width bound

Only the Gaussian sigma upper bound changes, from 4 to 10 s; mu remains between -1 and 7 s, and sigma starts at 0.05 s. Every observed training bin enters the exact count-weighted compressed fit, while the five held-out folds, bootstrap trial draws, group exponential predictions and flat residual variance remain the saved ones. Positive contrast favours the Gaussian. Tests are two-sided Wilcoxon tests of the 15 session medians, with raw p-values.

| group | bound_s | n | sessions | sigma_median | sigma_q25 | sigma_q75 | at_4_pct | at_10_pct | gaussian_better | exponential_better | undecided | median_session_contrast | wilcoxon_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tic_down | 4 | 404 | 15 | 2.3427 | 0.85016 | 4 | 41.089 | 0 | 6.6832 | 17.079 | 76.238 | -0.00066147 | 0.015076 |
| tic_down | 10 | 404 | 15 | 2.4607 | 0.87923 | 10 | 0 | 36.139 | 7.1782 | 16.089 | 76.733 | -0.00056008 | 0.018066 |
| tic_up | 4 | 479 | 15 | 2.4779 | 1.5827 | 4 | 29.019 | 0 | 15.031 | 3.9666 | 81.002 | 0.001017 | 0.10699 |
| tic_up | 10 | 479 | 15 | 2.5084 | 1.5924 | 4.9085 | 0 | 17.537 | 15.031 | 3.5491 | 81.42 | 0.0011518 | 0.12054 |
| time_cell | 4 | 1177 | 15 | 1.0596 | 0.71827 | 1.4626 | 2.0391 | 0 | 26.678 | 6.4571 | 66.865 | 0.00096127 | 6.1035e-05 |
| time_cell | 10 | 1177 | 15 | 1.0611 | 0.71934 | 1.4625 | 0 | 1.5293 | 26.763 | 6.4571 | 66.78 | 0.00096052 | 6.1035e-05 |

Fit failures: 1. Wall time: 179.9 s. Bound flags use absolute tolerance 1e-5. Full-data widths describe the same cells as the held-out contrasts; local nonlinear refinement is the weakest numerical step.
