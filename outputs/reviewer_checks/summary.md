# wider-window reviewer checks

The model comparisons contain the same 404 ramp-down TICs, 479 ramp-up TICs and 1,177 time cells across 15 sessions, with no trial-bin subsampling; the threshold analysis reads all 5,161 labelled cells and all their source trials. The saved folds, bootstrap draws, held-out reference predictions and flat residual variances are reused. The six analyses have one script each, with the fit-quality extension selected by analysis/fit_quality.py --two-exponential.

## run time and verification

| analysis | seconds |
| --- | --- |
| 1. two-exponential bump | 633.03 |
| 2. equal time-bin weights | 110.88 |
| 3. Gaussian width bound | 179.86 |
| 4. single-trial threshold | 417.5 |
| 5. onset, relaxation and peak-width correlations | 2.4 |
| 6. extended fit examples | 27.609 |

The runs overlapped, so these wall times should not be added to estimate elapsed time. The final test suite passed 49 tests in 44.25 s, with no test expectations changed for layout. The new scientific figure is figures/fit_examples_two_exponential.png and .pdf, 13 by 14.5 inches, with a 14 pt minimum font (7 pt after scaling to 6.5 inches wide). Visual review of the PNG and a PDF rendering found no remaining overlap or clipping, and pdffonts reports embedded CID TrueType fonts. Long example-axis labels were wrapped, and the three box labels were spaced and rotated to avoid touching.

## fit failures and input observations

Two-exponential: two fits did not converge, both for A132-20221124-04 neuron 60, last anchoring (fold 2 and full data), out of 24,720 fits. The width-bound check has one non-converged Gaussian fit, A132-20221124-04 neuron 386, fold 3, out of 12,360 fits. Their returned parameter estimates and predictions remain in the comparisons, with failure rows retained; the code does not silently drop these cells.

Equal weighting: no attempted fit failed to converge, but 75 rise fits in A215-20251013-02 are unsupported, comprising 27 ramp-down TICs, five ramp-up TICs and 43 time cells; its 125 saved trials leave only two exact rise-axis coordinates with at least 10 contributing trials, insufficient to identify four parameters. The 6,105 attempted fits are summarised on their matched defined subsets. This follows the exact-coordinate compression convention rather than rounding the next-lick axis to create support.

The raw label table contains 106 cells labelled both TIC and time cell, reported separately; all TIC summaries include them (n=989). Four saved TIC labels fail the recomputed all-trial rule:

| session | neu_id | group | is_tic | full_pass |
| --- | --- | --- | --- | --- |
| A132-20221124-02 | 415 | tic_up | True | False |
| A133-20230120-06 | 97 | tic_and_time_cell | True | False |
| A133-20230114-02 | 204 | tic_up | True | False |
| A145-20230313-02 | 119 | tic_down | True | False |

The endpoint and peak requirements conflict when p is zero or T; the explicit endpoint tests take precedence, so both endpoint values are baseline, while the peak equals a+B for an interior p. All degenerate cases remain finite.

The time-cell bump fit is unexpectedly weak (median averaged-trace R2=0.156). After seeing it, the 12 fixed example fits were checked against their best coarse-grid objectives, without changing a fit; every refinement was at least as good within 6e-9 SSE. The prescribed falling branch has negative curvature for positive B, whereas a decaying exponential has positive curvature, which is a concrete shape limitation; the check does not prove global optimality.

The first bump figure attempt began before all session fits were available and stopped; it was rerun after completion, then its duplicated alignment code was replaced with the original averaging helper so exact interpolation endpoints use the same operator for every model. The retained Gaussian and exponential R2 values agree with the original CSV to absolute tolerance 1e-12, and all 12 example IDs are unchanged. The original fit-quality figures and outputs are preserved. An accidental rerun of the three earlier figure commands was corrected by restoring those figure pairs from HEAD; the deterministic older summary-table commands were also rerun, and no new statistical variant from those commands is retained in this commit.

Held-out bump predictions and intermediate per-session results remain locally under outputs/two_exponential; the commit contains the requested per-cell tables, all summaries and failure records, while large held-out arrays and the per-cell random-removal draws remain generated local data.

## 1. two-exponential bump

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

## 2. equal time-bin weights

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

## 3. Gaussian width bound

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

## 4. single-trial threshold

All raw trials enter, with the event at sample 1500 and 500 Hz sampling. Each trial uses its cell's complete trial-average minimum and maximum, separately for last-lick and next-lick arrays; the threshold differences retain their definition when a trial's normalised pre-event mean is zero or negative. The pre-event window is 1250:1500 and the post-event window is 1500:1750.

| group | threshold | n | defined | median | q25 | q75 | pct_cells_above_50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tic_down | last | 404 | 404 | 58.869 | 54.483 | 63.855 | 94.059 |
| tic_down | next | 404 | 404 | 61.268 | 56.232 | 67.133 | 96.782 |
| tic_down | both | 404 | 404 | 36.545 | 32.067 | 44.277 | 14.109 |
| tic_up | last | 479 | 479 | 56.461 | 53.012 | 60.428 | 91.858 |
| tic_up | next | 479 | 479 | 60.123 | 55.215 | 66.705 | 96.033 |
| tic_up | both | 479 | 479 | 35.398 | 30.769 | 40.767 | 6.4718 |
| time_cell | last | 1177 | 1177 | 50 | 42.331 | 58.193 | 49.618 |
| time_cell | next | 1177 | 1177 | 46.479 | 37.209 | 54.93 | 39.677 |
| time_cell | both | 1177 | 1177 | 24.648 | 17.606 | 31.132 | 2.6338 |
| other | last | 2995 | 2995 | 46.032 | 39.683 | 51.724 | 31.052 |
| other | next | 2995 | 2995 | 49.157 | 41.15 | 56.024 | 46.277 |
| other | both | 2995 | 2995 | 23.256 | 17.699 | 28.571 | 2.2705 |
| tic_and_time_cell | last | 106 | 106 | 59.244 | 55.748 | 63.813 | 95.283 |
| tic_and_time_cell | next | 106 | 106 | 59.084 | 56.011 | 64.021 | 97.17 |
| tic_and_time_cell | both | 106 | 106 | 37.377 | 32.36 | 42.212 | 4.717 |
| all_TICs | last | 989 | 989 | 57.394 | 53.782 | 62.048 | 93.124 |
| all_TICs | next | 989 | 989 | 60.504 | 55.801 | 66.667 | 96.461 |
| all_TICs | both | 989 | 989 | 36.145 | 31.338 | 42.169 | 9.4034 |

The trial-removal denominator is all saved TICs, including the separately reported TIC/time-cell overlap group. Top trials are ranked only on last-lick samples 1500:1750, with ties in row order; ceil(k times n) paired trial rows are removed from both arrays, and each remaining average is normalised anew. The null uses the same count with 200 independent random removals per cell, with seed 20261116 and paired rows across alignments; each repeat aggregates cells before calculating the reported mean and 95% range.

| group | drop_pct | n | top_retained_pct | random_mean_pct | random_q025 | random_q975 | repeats |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tic_down | 10 | 404 | 79.208 | 91.699 | 89.851 | 93.564 | 200 |
| tic_down | 20 | 404 | 59.901 | 86.969 | 84.406 | 89.858 | 200 |
| tic_down | 30 | 404 | 40.099 | 82.749 | 79.944 | 85.644 | 200 |
| tic_up | 10 | 479 | 65.553 | 91.075 | 88.727 | 93.116 | 200 |
| tic_up | 20 | 479 | 41.962 | 85.951 | 83.293 | 88.727 | 200 |
| tic_up | 30 | 479 | 27.14 | 81.632 | 79.118 | 84.139 | 200 |
| tic_and_time_cell | 10 | 106 | 74.528 | 85.764 | 80.189 | 90.59 | 200 |
| tic_and_time_cell | 20 | 106 | 58.491 | 79.231 | 72.642 | 85.849 | 200 |
| tic_and_time_cell | 30 | 106 | 40.566 | 73.387 | 66.981 | 81.132 | 200 |
| all_TICs | 10 | 989 | 72.093 | 90.761 | 89.378 | 92.113 | 200 |
| all_TICs | 20 | 989 | 51.062 | 85.647 | 83.718 | 87.568 | 200 |
| all_TICs | 30 | 989 | 33.873 | 81.204 | 79.171 | 83.42 | 200 |

Input observations: 4 (details in issues.csv). The labels include 106 TIC/time-cell overlaps, retained separately. Existing label_stability drop_top uses 1250:1750 and is left intact. Wall time: 417.5 s. The trials are paired by their source row, as in the original rule; matching row identities across the two source arrays is the weakest input assumption.

## 5. onset, relaxation and peak-width correlations

Both parameters must be away from either saved bound; exclusion uses the wider-window bound flags, with no refitting. Spearman correlations are computed across retained cells and separately within each session; raw two-sided Wilcoxon p-values test the defined session correlations against zero. Rise mu retains its saved time-before-next-lick definition. IQRs describe variation between sessions, not uncertainty in a pooled coefficient.

| group | x | y | n | excluded | pooled_r | sessions_defined | median_session_r | session_r_q25 | session_r_q75 | wilcoxon_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tic_down | decay_mu | decay_tau | 207 | 197 | -0.28875 | 15 | -0.25275 | -0.48571 | -0.079798 | 0.012451 |
| tic_up | rise_mu | rise_tau | 282 | 197 | -0.57879 | 15 | -0.6 | -0.67404 | -0.44761 | 0.00042725 |
| time_cell | decay_mu | decay_tau | 735 | 442 | -0.17754 | 15 | -0.35743 | -0.41282 | -0.031962 | 0.0083618 |
| time_cell | gaussian_mu | gaussian_sigma | 1080 | 97 | -0.042196 | 15 | -0.046931 | -0.21301 | 0.08468 | 0.20776 |

Wall time: 2.4 s. Excluding bounded cells can alter the sampled relationship; the pooled and session estimates are reported together so a heavily represented session can be recognised.

## 6. extended fit examples

The original Gaussian and exponential R2 values and the 12 saved example IDs are retained exactly, while the bump uses the same trial-bin mask, interpolation operator and minimum of 10 trials per averaged bin. Both ramp-up alignments are retained in the table; the figure uses next-lick alignment with negative time increasing towards zero. All curves are full-data fits, so these R2 values describe the fitted averages rather than held-out prediction.

| group | alignment | model | n | defined | median | q25 | q75 | above_half_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tic_down | last_lick | gaussian | 404 | 404 | 0.64522 | 0.45076 | 0.79616 | 68.812 |
| tic_down | last_lick | exponential | 404 | 404 | 0.61383 | 0.43398 | 0.75225 | 66.089 |
| tic_down | last_lick | bump | 404 | 404 | 0.64782 | 0.45182 | 0.80919 | 69.059 |
| tic_up | last_lick | gaussian | 479 | 479 | 0.78221 | 0.6464 | 0.88175 | 87.056 |
| tic_up | last_lick | exponential | 479 | 479 | 0.59217 | 0.41752 | 0.73902 | 63.883 |
| tic_up | last_lick | bump | 479 | 479 | 0.68077 | 0.47837 | 0.81417 | 73.278 |
| tic_up | next_lick | gaussian | 479 | 479 | 0.72492 | 0.53809 | 0.84791 | 77.662 |
| tic_up | next_lick | exponential | 479 | 479 | 0.79335 | 0.62491 | 0.89792 | 84.969 |
| tic_up | next_lick | bump | 479 | 479 | 0.79856 | 0.6255 | 0.89597 | 85.177 |
| time_cell | last_lick | gaussian | 1177 | 1177 | 0.87655 | 0.68438 | 0.94949 | 87.171 |
| time_cell | last_lick | exponential | 1177 | 1177 | 0.72605 | 0.54925 | 0.8475 | 79.779 |
| time_cell | last_lick | bump | 1177 | 1177 | 0.15617 | -0.0101 | 0.43978 | 20.221 |

| group | median_session_contrast | wilcoxon_p |
| --- | --- | --- |
| tic_down | 4.0857e-05 | 0.71973 |
| tic_up | -0.00022266 | 0.0083618 |
| time_cell | 3.925e-16 | 0.10974 |

The primary anchoring is selected by the sign of the median session contrast: tic_down: last, tic_up: next, time_cell: last. The down and time-cell anchoring comparisons are not significant, and the time-cell median contrast is numerically close to zero.

| session | neu_id | group | alignment | percentile | r2_gaussian | r2_exponential | r2_bump | bump_anchoring |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A215-20251010-04 | 269 | tic_down | last_lick | 20 | 0.53147 | 0.53295 | 0.71862 | last |
| A163-20230903-02 | 103 | tic_down | last_lick | 40 | 0.24895 | 0.67373 | 0.46528 | last |
| A147-20230401-02 | 204 | tic_down | last_lick | 60 | 0.78353 | 0.61346 | 0.82676 | last |
| A133-20230114-02 | 88 | tic_down | last_lick | 80 | 0.8496 | 0.85088 | 0.93451 | last |
| A164-20230902-05 | 411 | tic_up | next_lick | 20 | 0.43661 | 0.6083 | 0.61353 | next |
| A133-20230120-06 | 11 | tic_up | next_lick | 40 | 0.71542 | 0.76914 | 0.77596 | next |
| A213-20250916-04 | 167 | tic_up | next_lick | 60 | 0.86277 | 0.83922 | 0.87779 | next |
| A163-20230903-02 | 165 | tic_up | next_lick | 80 | 0.84005 | 0.92369 | 0.87047 | next |
| A214-20251009-04 | 277 | time_cell | last_lick | 20 | 0.59241 | 0.63753 | 0.41489 | last |
| A164-20230902-05 | 74 | time_cell | last_lick | 40 | 0.83247 | 0.74883 | 0.083282 | last |
| A132-20221124-02 | 112 | time_cell | last_lick | 60 | 0.9158 | 0.83814 | 0.32715 | last |
| A143-20230315-02 | 144 | time_cell | last_lick | 80 | 0.96 | 0.9013 | 0.22946 | last |

Canvas: 13 by 14.5 inches; smallest font: 14 pt (7 pt at 6.5 inches wide). Original outputs and figures are preserved.

Wall time: 27.6 s.

The first figure attempt began before every bump parameter file was present and stopped without completing; it was rerun after the full fit finished, then the extension was moved into analysis/fit_quality.py with the --two-exponential option. Visual review found clipped long example-axis labels and touching box labels, corrected by wrapping the axis text, spacing the boxes and rotating their labels; all fonts remain at least 14 pt.
