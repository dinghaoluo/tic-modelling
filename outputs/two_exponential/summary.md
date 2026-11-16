# two-exponential bumps

All observed trial bins enter the fit, with identical (t, T) pairs compressed to their means and weighted by count; the coarse grid profiles a and non-negative B before bounded least squares. The saved five folds, trial bootstrap draws and flat residual variance are retained, and t >= 0 is scored without refitting. Positive contrast favours model_a; the last-versus-next row is (error_next minus error_last) / (2 variance). All Wilcoxon tests use the 15 session medians and raw two-sided p-values.

| group | model_a | model_b | scoring | n | sessions | better | worse | undecided | median_session_contrast | wilcoxon_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tic_down | last | decay | all | 404 | 15 | 19.802 | 2.9703 | 77.228 | 0.00056953 | 0.00061035 |
| tic_down | last | decay | post_last_lick | 404 | 15 | 17.574 | 2.2277 | 80.198 | 0.00059557 | 0.0015259 |
| tic_down | last | gaussian | all | 404 | 15 | 24.505 | 1.4851 | 74.01 | 0.0016094 | 6.1035e-05 |
| tic_down | last | gaussian | post_last_lick | 404 | 15 | 19.802 | 1.2376 | 78.96 | 0.0014265 | 6.1035e-05 |
| tic_down | last | next | all | 404 | 15 | 5.9406 | 4.703 | 89.356 | 4.0857e-05 | 0.71973 |
| tic_down | last | next | post_last_lick | 404 | 15 | 5.9406 | 4.9505 | 89.109 | 6.2798e-06 | 0.59949 |
| tic_down | next | decay | all | 404 | 15 | 14.604 | 1.7327 | 83.663 | 0.00052042 | 0.010254 |
| tic_down | next | decay | post_last_lick | 404 | 15 | 11.881 | 1.4851 | 86.634 | 0.00048143 | 0.021545 |
| tic_down | next | gaussian | all | 404 | 15 | 20.545 | 1.4851 | 77.97 | 0.0011885 | 6.1035e-05 |
| tic_down | next | gaussian | post_last_lick | 404 | 15 | 17.079 | 2.2277 | 80.693 | 0.0011395 | 0.00018311 |
| tic_up | last | gaussian | all | 479 | 15 | 13.57 | 1.6701 | 84.76 | 0.0011773 | 0.0026245 |
| tic_up | last | gaussian | post_last_lick | 479 | 15 | 10.021 | 3.3403 | 86.639 | 0.0013365 | 0.0083618 |
| tic_up | last | next | all | 479 | 15 | 5.428 | 8.5595 | 86.013 | -0.00022266 | 0.0083618 |
| tic_up | last | next | post_last_lick | 479 | 15 | 6.0543 | 7.5157 | 86.43 | -0.00018412 | 0.012451 |
| tic_up | last | rise | all | 479 | 15 | 19.833 | 0.83507 | 79.332 | 0.0020769 | 0.00042725 |
| tic_up | last | rise | post_last_lick | 479 | 15 | 10.856 | 2.0877 | 87.056 | 0.0011831 | 0.018066 |
| tic_up | next | gaussian | all | 479 | 15 | 12.944 | 1.0438 | 86.013 | 0.00096002 | 0.00085449 |
| tic_up | next | gaussian | post_last_lick | 479 | 15 | 9.6033 | 1.8789 | 88.518 | 0.0012321 | 0.0067139 |
| tic_up | next | rise | all | 479 | 15 | 24.635 | 1.0438 | 74.322 | 0.0017313 | 6.1035e-05 |
| tic_up | next | rise | post_last_lick | 479 | 15 | 16.91 | 0.83507 | 82.255 | 0.0015475 | 0.0011597 |
| time_cell | last | decay | all | 1177 | 15 | 4.6729 | 26.933 | 68.394 | -0.0031481 | 0.00012207 |
| time_cell | last | decay | post_last_lick | 1177 | 15 | 6.8819 | 27.188 | 65.93 | -0.002452 | 0.00012207 |
| time_cell | last | gaussian | all | 1177 | 15 | 3.9932 | 32.285 | 63.721 | -0.0050175 | 6.1035e-05 |
| time_cell | last | gaussian | post_last_lick | 1177 | 15 | 5.6924 | 27.273 | 67.035 | -0.0038813 | 0.00030518 |
| time_cell | last | next | all | 1177 | 15 | 5.5225 | 7.7315 | 86.746 | 3.925e-16 | 0.10974 |
| time_cell | last | next | post_last_lick | 1177 | 15 | 9.4308 | 9.7706 | 80.799 | 2.1556e-14 | 0.10951 |
| time_cell | next | decay | all | 1177 | 15 | 4.503 | 26.593 | 68.904 | -0.0038349 | 0.00012207 |
| time_cell | next | decay | post_last_lick | 1177 | 15 | 5.1827 | 27.273 | 67.545 | -0.0037222 | 0.00012207 |
| time_cell | next | gaussian | all | 1177 | 15 | 3.4834 | 31.436 | 65.081 | -0.0061554 | 6.1035e-05 |
| time_cell | next | gaussian | post_last_lick | 1177 | 15 | 5.6924 | 27.273 | 67.035 | -0.0041871 | 0.00030518 |

| group | anchoring | parameter | n | median | q25 | q75 | lower_bound_pct | upper_bound_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tic_down | last | u | 404 | 1.624 | 0.845 | 2.7394 | 6.1881 | 3.4653 |
| tic_down | last | tau1 | 404 | 0.45246 | 0.11179 | 2.6607 | 22.525 | 16.832 |
| tic_down | last | tau2 | 404 | 1.0528 | 0.15947 | 10 | 14.356 | 34.653 |
| tic_down | next | u | 404 | 2.3295 | 0.95099 | 3.3805 | 4.2079 | 6.9307 |
| tic_down | next | tau1 | 404 | 0.59369 | 0.22398 | 3.2608 | 14.356 | 20.297 |
| tic_down | next | tau2 | 404 | 0.40854 | 0.11421 | 10 | 21.287 | 25.743 |
| tic_up | last | u | 479 | 4.2014 | 2.8671 | 5.6272 | 0.6263 | 22.756 |
| tic_up | last | tau1 | 479 | 10 | 2.3054 | 10 | 1.6701 | 57.829 |
| tic_up | last | tau2 | 479 | 0.11048 | 0.1 | 0.55748 | 45.511 | 17.119 |
| tic_up | next | u | 479 | 0.50965 | 0.055059 | 1.726 | 20.668 | 0 |
| tic_up | next | tau1 | 479 | 9.9978 | 1.78 | 10 | 0.41754 | 48.643 |
| tic_up | next | tau2 | 479 | 0.1 | 0.1 | 0.21138 | 52.192 | 3.9666 |
| time_cell | last | u | 1177 | 0.023981 | 1e-10 | 0.74777 | 42.481 | 1.5293 |
| time_cell | last | tau1 | 1177 | 0.1 | 0.1 | 0.1657 | 71.963 | 13.424 |
| time_cell | last | tau2 | 1177 | 10 | 0.75467 | 10 | 23.364 | 69.584 |
| time_cell | next | u | 1177 | 4.242 | 0.79008 | 7 | 21.58 | 32.71 |
| time_cell | next | tau1 | 1177 | 0.1 | 0.1 | 0.26565 | 67.46 | 14.274 |
| time_cell | next | tau2 | 1177 | 10 | 0.37828 | 10 | 23.28 | 72.133 |

Endpoint values are defined as baseline even when the peak coincides with an endpoint; this is the continuous interior formula with a baseline endpoint convention, since the stated peak and baseline constraints conflict at p=0 or p=T. B has lower bound zero and no upper bound. Bound flags use absolute tolerance 1e-5.

tic_down: session medians favour last anchoring (p=0.71973); this is not significant.

tic_up: session medians favour next anchoring (p=0.0083618); this is significant.

time_cell: session medians favour last anchoring (p=0.10974); this is not significant.

After seeing the low time-cell R2, the 12 saved example fits were checked against their best coarse-grid objective without changing any fit; every refined objective was at least as good within 6e-9 SSE. The prescribed falling branch has negative curvature for positive B, whereas an ordinary decaying exponential has positive curvature. This check does not establish a global optimum; details are in example_objective_check.csv.

Fit failures: 2 of 24720 fits. Wall time: 633.0 s. Parameter IQRs describe cells, while trial-bootstrap intervals describe each cell's held-out contrast. Local minima remain the weakest step; both anchorings use the same coarse grid and bounded refinement.
