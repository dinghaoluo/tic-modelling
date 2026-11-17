# revised bumps and plain exponentials

The original data, five folds, trial bootstrap resamples, flat residual variance, contrast sign and 95% decision interval are retained. All observed trial bins enter each fit, with identical (t, T) pairs averaged and weighted by count, without subsampling. All-bin scoring is primary; t >= 0 is scored from the same held-out predictions without refitting. Positive contrasts favour model_a. Wilcoxon p-values are raw two-sided tests of the 15 session medians against zero. The first bump always uses last anchoring for ramp-down TICs and time cells, and next anchoring for ramp-up TICs.

Both revised anchorings are fitted for every cell. A coarse grid (t0: -1, 0, 1, 2, 3 s; u: 0, 0.5, 1, 2, 3, 4, 5.5, 7 s; four logarithmic values per tau) profiles a and non-negative B, followed by bounded least squares with the profiled analytic Jacobian. The plain model uses 10 evenly spaced mu and 12 logarithmic tau starts, matching the existing exponential grid density. The plain model is exactly a + B exp(-max(x - mu, 0) / tau), with x = T - t for ramp-up TICs and x = t otherwise.

| group | model_a | model_b | scoring | n | sessions | better | worse | undecided | median_session_contrast | wilcoxon_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tic_down | plain | decay | all | 404 | 15 | 2.7228 | 37.871 | 59.406 | -0.0034305 | 0.00042725 |
| tic_down | plain | decay | post_last_lick | 404 | 15 | 7.6733 | 2.2277 | 90.099 | -0.00028967 | 0.3028 |
| tic_down | plain | first_bump | all | 404 | 15 | 0.49505 | 39.356 | 60.149 | -0.0045924 | 6.1035e-05 |
| tic_down | plain | first_bump | post_last_lick | 404 | 15 | 3.9604 | 5.198 | 90.842 | -0.00099727 | 0.0042725 |
| tic_down | plain | gaussian | all | 404 | 15 | 7.6733 | 35.149 | 57.178 | -0.0024627 | 6.1035e-05 |
| tic_down | plain | gaussian | post_last_lick | 404 | 15 | 13.861 | 1.2376 | 84.901 | 0.00021142 | 0.4212 |
| tic_down | revised_last | decay | all | 404 | 15 | 16.584 | 4.703 | 78.713 | 7.8215e-05 | 0.12054 |
| tic_down | revised_last | decay | post_last_lick | 404 | 15 | 14.604 | 3.9604 | 81.436 | 0.00022821 | 0.083252 |
| tic_down | revised_last | first_bump | all | 404 | 15 | 5.198 | 11.386 | 83.416 | -0.00010885 | 0.021545 |
| tic_down | revised_last | first_bump | post_last_lick | 404 | 15 | 4.4554 | 9.4059 | 86.139 | -0.00014108 | 0.0067139 |
| tic_down | revised_last | gaussian | all | 404 | 15 | 21.287 | 0.9901 | 77.723 | 0.00078169 | 0.00030518 |
| tic_down | revised_last | gaussian | post_last_lick | 404 | 15 | 18.564 | 1.2376 | 80.198 | 0.00091156 | 6.1035e-05 |
| tic_down | revised_next | decay | all | 404 | 15 | 10.644 | 5.4455 | 83.911 | 0.00014532 | 0.5614 |
| tic_down | revised_next | decay | post_last_lick | 404 | 15 | 10.149 | 2.7228 | 87.129 | 0.00023101 | 0.63867 |
| tic_down | revised_next | first_bump | all | 404 | 15 | 2.4752 | 11.386 | 86.139 | -0.00024153 | 0.0053711 |
| tic_down | revised_next | first_bump | post_last_lick | 404 | 15 | 2.9703 | 6.9307 | 90.099 | -0.00022768 | 0.12054 |
| tic_down | revised_next | gaussian | all | 404 | 15 | 14.109 | 1.7327 | 84.158 | 0.001161 | 0.00061035 |
| tic_down | revised_next | gaussian | post_last_lick | 404 | 15 | 12.871 | 1.9802 | 85.149 | 0.00094931 | 0.0015259 |
| tic_up | plain | first_bump | all | 479 | 15 | 0.41754 | 21.503 | 78.079 | -0.001488 | 6.1035e-05 |
| tic_up | plain | first_bump | post_last_lick | 479 | 15 | 1.2526 | 16.701 | 82.046 | -0.0013041 | 0.0026245 |
| tic_up | plain | gaussian | all | 479 | 15 | 4.3841 | 12.109 | 83.507 | -0.00028989 | 0.20776 |
| tic_up | plain | gaussian | post_last_lick | 479 | 15 | 4.3841 | 8.3507 | 87.265 | -0.00011447 | 0.80396 |
| tic_up | plain | rise | all | 479 | 15 | 6.6806 | 13.152 | 80.167 | -0.00020168 | 0.25238 |
| tic_up | plain | rise | post_last_lick | 479 | 15 | 3.9666 | 10.856 | 85.177 | -0.00017038 | 0.3894 |
| tic_up | revised_last | first_bump | all | 479 | 15 | 1.4614 | 12.109 | 86.43 | -0.00051649 | 6.1035e-05 |
| tic_up | revised_last | first_bump | post_last_lick | 479 | 15 | 3.3403 | 8.3507 | 88.309 | -0.00038241 | 0.00085449 |
| tic_up | revised_last | gaussian | all | 479 | 15 | 12.317 | 5.6367 | 82.046 | 0.00039991 | 0.094604 |
| tic_up | revised_last | gaussian | post_last_lick | 479 | 15 | 9.6033 | 5.6367 | 84.76 | 0.00030555 | 0.3028 |
| tic_up | revised_last | rise | all | 479 | 15 | 17.537 | 1.8789 | 80.585 | 0.0015437 | 0.015076 |
| tic_up | revised_last | rise | post_last_lick | 479 | 15 | 9.1858 | 3.1315 | 87.683 | 0.00051852 | 0.047913 |
| tic_up | revised_next | first_bump | all | 479 | 15 | 1.6701 | 16.075 | 82.255 | -0.00028448 | 0.035339 |
| tic_up | revised_next | first_bump | post_last_lick | 479 | 15 | 4.1754 | 11.273 | 84.551 | -0.00020537 | 0.0083618 |
| tic_up | revised_next | gaussian | all | 479 | 15 | 12.526 | 7.0981 | 80.376 | 0.00025193 | 0.27686 |
| tic_up | revised_next | gaussian | post_last_lick | 479 | 15 | 8.977 | 7.5157 | 83.507 | 5.5709e-05 | 0.89038 |
| tic_up | revised_next | rise | all | 479 | 15 | 19.833 | 3.3403 | 76.827 | 0.0013971 | 6.1035e-05 |
| tic_up | revised_next | rise | post_last_lick | 479 | 15 | 12.109 | 1.6701 | 86.221 | 0.00062109 | 0.0042725 |
| time_cell | plain | decay | all | 1177 | 15 | 21.325 | 8.4112 | 70.263 | 0.00025832 | 0.072998 |
| time_cell | plain | decay | post_last_lick | 1177 | 15 | 8.836 | 5.3526 | 85.811 | 8.5918e-05 | 0.094604 |
| time_cell | plain | first_bump | all | 1177 | 15 | 26.253 | 5.8624 | 67.884 | 0.0029774 | 6.1035e-05 |
| time_cell | plain | first_bump | post_last_lick | 1177 | 15 | 26.848 | 2.6338 | 70.518 | 0.0023324 | 0.00012207 |
| time_cell | plain | gaussian | all | 1177 | 15 | 9.0059 | 16.822 | 74.172 | -0.00050402 | 0.0042725 |
| time_cell | plain | gaussian | post_last_lick | 1177 | 15 | 8.7511 | 5.6924 | 85.556 | -8.1153e-05 | 0.52448 |
| time_cell | revised_last | decay | all | 1177 | 15 | 11.895 | 12.404 | 75.701 | -0.00074602 | 0.083252 |
| time_cell | revised_last | decay | post_last_lick | 1177 | 15 | 8.836 | 17.332 | 73.832 | -0.0012458 | 0.0053711 |
| time_cell | revised_last | first_bump | all | 1177 | 15 | 35.004 | 4.3331 | 60.663 | 0.003655 | 6.1035e-05 |
| time_cell | revised_last | first_bump | post_last_lick | 1177 | 15 | 28.377 | 3.9082 | 67.715 | 0.0024653 | 0.00012207 |
| time_cell | revised_last | gaussian | all | 1177 | 15 | 5.0127 | 18.607 | 76.381 | -0.001562 | 0.00030518 |
| time_cell | revised_last | gaussian | post_last_lick | 1177 | 15 | 5.9473 | 15.208 | 78.845 | -0.0013034 | 0.0026245 |
| time_cell | revised_next | decay | all | 1177 | 15 | 5.6075 | 11.215 | 83.178 | -0.00177 | 0.0015259 |
| time_cell | revised_next | decay | post_last_lick | 1177 | 15 | 5.3526 | 16.737 | 77.91 | -0.0026705 | 0.0020142 |
| time_cell | revised_next | first_bump | all | 1177 | 15 | 25.913 | 4.5879 | 69.499 | 0.0020306 | 0.00018311 |
| time_cell | revised_next | first_bump | post_last_lick | 1177 | 15 | 15.548 | 4.418 | 80.034 | 0.00084325 | 0.083252 |
| time_cell | revised_next | gaussian | all | 1177 | 15 | 4.418 | 17.587 | 77.995 | -0.0029935 | 0.00018311 |
| time_cell | revised_next | gaussian | post_last_lick | 1177 | 15 | 5.5225 | 15.633 | 78.845 | -0.0026756 | 0.00085449 |

## full-data parameters

| group | model | parameter | n | median | q25 | q75 | lower_bound_pct | upper_bound_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tic_down | plain | a | 404 | 0.027798 | -0.0039093 | 0.050714 | nan | nan |
| tic_down | plain | B | 404 | 0.0029398 | 0 | 0.10185 | 48.02 | nan |
| tic_down | plain | mu | 404 | 1.1486 | -1 | 3 | 48.02 | 28.465 |
| tic_down | plain | tau | 404 | 0.1 | 0.1 | 2.6928 | 56.188 | 20.545 |
| tic_down | revised_last | a | 404 | -0.01391 | -0.03318 | 0.0034528 | nan | nan |
| tic_down | revised_last | B | 404 | 0.10379 | 0.060697 | 0.18291 | 0 | nan |
| tic_down | revised_last | t0 | 404 | -0.19244 | -0.55485 | -0.035183 | 17.079 | 2.2277 |
| tic_down | revised_last | u | 404 | 1.7763 | 0.87409 | 2.9449 | 3.2178 | 2.4752 |
| tic_down | revised_last | tau1 | 404 | 1.468 | 0.37468 | 10 | 8.4158 | 26.733 |
| tic_down | revised_last | tau2 | 404 | 0.84604 | 0.14725 | 10 | 15.099 | 31.436 |
| tic_down | revised_next | a | 404 | -0.013495 | -0.034537 | 0.0069589 | nan | nan |
| tic_down | revised_next | B | 404 | 0.10261 | 0.064264 | 0.17438 | 0 | nan |
| tic_down | revised_next | t0 | 404 | -0.06269 | -0.36717 | 0.034595 | 10.396 | 0.49505 |
| tic_down | revised_next | u | 404 | 2.259 | 0.84006 | 3.3153 | 3.7129 | 4.9505 |
| tic_down | revised_next | tau1 | 404 | 0.76482 | 0.23327 | 7.1637 | 10.149 | 22.525 |
| tic_down | revised_next | tau2 | 404 | 0.34415 | 0.10276 | 9.9846 | 23.267 | 24.257 |
| tic_up | plain | a | 479 | -0.059755 | -0.1578 | -0.0076651 | nan | nan |
| tic_up | plain | B | 479 | 0.15334 | 0.078374 | 0.27728 | 0.20877 | nan |
| tic_up | plain | mu | 479 | 1.7297 | 1.031 | 2.655 | 0.6263 | 18.789 |
| tic_up | plain | tau | 479 | 2.4621 | 1.0394 | 9.9437 | 1.2526 | 24.635 |
| tic_up | revised_last | a | 479 | -0.015238 | -0.038917 | -0.00039593 | nan | nan |
| tic_up | revised_last | B | 479 | 0.11308 | 0.071147 | 0.20052 | 0 | nan |
| tic_up | revised_last | t0 | 479 | -0.030426 | -0.31646 | 0.96164 | 15.866 | 4.3841 |
| tic_up | revised_last | u | 479 | 4.174 | 2.8953 | 5.5903 | 0.20877 | 17.328 |
| tic_up | revised_last | tau1 | 479 | 9.9883 | 1.9897 | 10 | 2.714 | 47.182 |
| tic_up | revised_last | tau2 | 479 | 0.13655 | 0.1 | 0.88714 | 41.754 | 18.163 |
| tic_up | revised_next | a | 479 | -0.015332 | -0.04149 | 0.00062194 | nan | nan |
| tic_up | revised_next | B | 479 | 0.11083 | 0.069985 | 0.18901 | 0 | nan |
| tic_up | revised_next | t0 | 479 | 0.073994 | -0.22607 | 1.0494 | 10.856 | 2.9228 |
| tic_up | revised_next | u | 479 | 0.68823 | 0.23359 | 1.9186 | 14.405 | 0 |
| tic_up | revised_next | tau1 | 479 | 5.5314 | 1.1968 | 10 | 1.0438 | 43.215 |
| tic_up | revised_next | tau2 | 479 | 0.10001 | 0.1 | 0.24716 | 49.478 | 6.6806 |
| time_cell | plain | a | 1177 | -0.051841 | -0.17679 | 0.009302 | nan | nan |
| time_cell | plain | B | 1177 | 0.22343 | 0.099615 | 0.43991 | 4.8428 | nan |
| time_cell | plain | mu | 1177 | 1.0575 | 0.18721 | 2.0055 | 6.2872 | 5.0977 |
| time_cell | plain | tau | 1177 | 0.91081 | 0.34681 | 1.6941 | 10.45 | 5.6075 |
| time_cell | revised_last | a | 1177 | -0.057722 | -0.15325 | -0.0064171 | nan | nan |
| time_cell | revised_last | B | 1177 | 0.24789 | 0.1237 | 0.4536 | 0 | nan |
| time_cell | revised_last | t0 | 1177 | -1 | -1 | -1 | 81.903 | 2.5489 |
| time_cell | revised_last | u | 1177 | 0.47352 | 1e-10 | 1.2022 | 34.919 | 1.1895 |
| time_cell | revised_last | tau1 | 1177 | 0.1 | 0.1 | 1.4786 | 51.402 | 15.123 |
| time_cell | revised_last | tau2 | 1177 | 10 | 10 | 10 | 2.7188 | 81.988 |
| time_cell | revised_next | a | 1177 | -0.042624 | -0.13036 | 0.0035454 | nan | nan |
| time_cell | revised_next | B | 1177 | 0.24303 | 0.12333 | 0.45045 | 0 | nan |
| time_cell | revised_next | t0 | 1177 | -0.99017 | -1 | -0.27599 | 47.239 | 1.8692 |
| time_cell | revised_next | u | 1177 | 5.2043 | 3.6971 | 7 | 2.6338 | 25.149 |
| time_cell | revised_next | tau1 | 1177 | 0.22344 | 0.1 | 1.333 | 37.553 | 19.116 |
| time_cell | revised_next | tau2 | 1177 | 10 | 10 | 10 | 4.418 | 82.668 |

The baseline a is unbounded; B has lower bound zero and no upper bound. Nonlinear bound flags use absolute tolerance 1e-5. IQRs describe variation between cells. When B is zero, nonlinear shape parameters are unidentifiable; these fits remain in the requested summaries and are flagged at the amplitude lower bound.

For the revised last anchoring, the effective trial onset is min(t0, p); clipping is counted once per cell and trial, not once per observed bin. For next anchoring, p = max(T - u, t0) already guarantees t0 <= p; if t0 > T the peak is beyond the trial and all observed bins are baseline. Empty pieces are omitted, and coincident onset/peak or peak/endpoint values use the original baseline endpoint convention.

| group | model | stage | fits | fits_clipped_pct | clipped_trial_pct | peak_beyond_endpoint_trial_pct |
| --- | --- | --- | --- | --- | --- | --- |
| tic_down | revised_last | 0 | 404 | 4.9505 | 2.2141 | 0 |
| tic_down | revised_last | 1 | 404 | 5.198 | 3.0615 | 0 |
| tic_down | revised_last | 2 | 404 | 3.9604 | 1.1426 | 0 |
| tic_down | revised_last | 3 | 404 | 4.703 | 2.3186 | 0 |
| tic_down | revised_last | 4 | 404 | 4.9505 | 1.8366 | 0 |
| tic_down | revised_last | full | 404 | 4.2079 | 2.0039 | 0 |
| tic_down | revised_next | 0 | 404 | 0 | 0 | 0.7603 |
| tic_down | revised_next | 1 | 404 | 0 | 0 | 0.53092 |
| tic_down | revised_next | 2 | 404 | 0 | 0 | 0.59827 |
| tic_down | revised_next | 3 | 404 | 0 | 0 | 0.56343 |
| tic_down | revised_next | 4 | 404 | 0 | 0 | 0.52521 |
| tic_down | revised_next | full | 404 | 0 | 0 | 0.48478 |
| tic_up | revised_last | 0 | 479 | 20.251 | 2.8581 | 0 |
| tic_up | revised_last | 1 | 479 | 21.086 | 4.3274 | 0 |
| tic_up | revised_last | 2 | 479 | 20.042 | 3.2139 | 0 |
| tic_up | revised_last | 3 | 479 | 21.294 | 3.4593 | 0 |
| tic_up | revised_last | 4 | 479 | 19.415 | 3.0932 | 0 |
| tic_up | revised_last | full | 479 | 18.372 | 2.939 | 0 |
| tic_up | revised_next | 0 | 479 | 0 | 0 | 1.9496 |
| tic_up | revised_next | 1 | 479 | 0 | 0 | 2.183 |
| tic_up | revised_next | 2 | 479 | 0 | 0 | 1.8439 |
| tic_up | revised_next | 3 | 479 | 0 | 0 | 2.1008 |
| tic_up | revised_next | 4 | 479 | 0 | 0 | 2.1242 |
| tic_up | revised_next | full | 479 | 0 | 0 | 1.8608 |
| time_cell | revised_last | 0 | 1177 | 7.3067 | 1.3143 | 0 |
| time_cell | revised_last | 1 | 1177 | 8.921 | 1.867 | 0 |
| time_cell | revised_last | 2 | 1177 | 7.7315 | 1.6782 | 0 |
| time_cell | revised_last | 3 | 1177 | 8.4962 | 1.8736 | 0 |
| time_cell | revised_last | 4 | 1177 | 7.6466 | 1.7305 | 0 |
| time_cell | revised_last | full | 1177 | 7.9864 | 1.5489 | 0 |
| time_cell | revised_next | 0 | 1177 | 0 | 0 | 0.52827 |
| time_cell | revised_next | 1 | 1177 | 0 | 0 | 0.44777 |
| time_cell | revised_next | 2 | 1177 | 0 | 0 | 0.42421 |
| time_cell | revised_next | 3 | 1177 | 0 | 0 | 0.46041 |
| time_cell | revised_next | 4 | 1177 | 0 | 0 | 0.41706 |
| time_cell | revised_next | full | 1177 | 0 | 0 | 0.44109 |

## fit failures

| session | neu_id | model | stage | issue |
| --- | --- | --- | --- | --- |
| A133-20230114-02 | 15 | revised_next | 3 | least_squares did not converge |
| A214-20251013-02 | 67 | revised_next | full | least_squares did not converge |

Fit failures: 2 of 37080 fits. Fitting wall time: 1000.4 s. Coarse-grid starts can still reach local minima; the grid and refinement are fixed before inspecting results.

The first completed session (A215-20251013-02, 75 cells) was reported before the full run, then rerun unchanged with the other sessions; its initial timing is retained in first_session_run.json. After inspecting the result, 48.0% of ramp-down plain fits were recognised as amplitude-bound fits, so their mu and tau summaries require the unidentifiability caveat above; no fits or exclusions were changed.

## averaged-trace R2

| group | alignment | model | n | defined | median | q25 | q75 | above_half_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tic_down | last_lick | gaussian | 404 | 404 | 0.64522 | 0.45076 | 0.79616 | 68.812 |
| tic_down | last_lick | exponential | 404 | 404 | 0.61383 | 0.43398 | 0.75225 | 66.089 |
| tic_down | last_lick | bump | 404 | 404 | 0.64782 | 0.45182 | 0.80919 | 69.059 |
| tic_down | last_lick | revised | 404 | 404 | 0.67912 | 0.4717 | 0.82688 | 70.792 |
| tic_down | last_lick | plain | 404 | 404 | -0.0031718 | -0.036519 | 0.15106 | 6.4356 |
| tic_up | last_lick | gaussian | 479 | 479 | 0.78221 | 0.6464 | 0.88175 | 87.056 |
| tic_up | last_lick | exponential | 479 | 479 | 0.59217 | 0.41752 | 0.73902 | 63.883 |
| tic_up | last_lick | bump | 479 | 479 | 0.68077 | 0.47837 | 0.81417 | 73.278 |
| tic_up | last_lick | revised | 479 | 479 | 0.70993 | 0.52449 | 0.82854 | 77.453 |
| tic_up | last_lick | plain | 479 | 479 | 0.6167 | 0.41068 | 0.75809 | 63.257 |
| tic_up | next_lick | gaussian | 479 | 479 | 0.72492 | 0.53809 | 0.84791 | 77.662 |
| tic_up | next_lick | exponential | 479 | 479 | 0.79335 | 0.62491 | 0.89792 | 84.969 |
| tic_up | next_lick | bump | 479 | 479 | 0.79856 | 0.6255 | 0.89597 | 85.177 |
| tic_up | next_lick | revised | 479 | 479 | 0.80477 | 0.61831 | 0.89957 | 85.177 |
| tic_up | next_lick | plain | 479 | 479 | 0.82566 | 0.67094 | 0.92012 | 88.1 |
| time_cell | last_lick | gaussian | 1177 | 1177 | 0.87655 | 0.68438 | 0.94949 | 87.171 |
| time_cell | last_lick | exponential | 1177 | 1177 | 0.72605 | 0.54925 | 0.8475 | 79.779 |
| time_cell | last_lick | bump | 1177 | 1177 | 0.15617 | -0.0101 | 0.43978 | 20.221 |
| time_cell | last_lick | revised | 1177 | 1177 | 0.73215 | 0.48936 | 0.84737 | 74.257 |
| time_cell | last_lick | plain | 1177 | 1177 | 0.73426 | 0.47003 | 0.89992 | 72.557 |

## unchanged example cells

| session | neu_id | group | alignment | percentile | revised_anchoring | revised_fit_success | plain_fit_success | r2_gaussian | r2_exponential | r2_bump | r2_revised | r2_plain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A215-20251010-04 | 269 | tic_down | last_lick | 20 | last | True | True | 0.53147 | 0.53295 | 0.71862 | 0.73354 | -8.1703e-07 |
| A163-20230903-02 | 103 | tic_down | last_lick | 40 | last | True | True | 0.24895 | 0.67373 | 0.46528 | 0.6353 | 0.0035918 |
| A147-20230401-02 | 204 | tic_down | last_lick | 60 | last | True | True | 0.78353 | 0.61346 | 0.82676 | 0.78915 | -0.056639 |
| A133-20230114-02 | 88 | tic_down | last_lick | 80 | last | True | True | 0.8496 | 0.85088 | 0.93451 | 0.93444 | 0.028821 |
| A164-20230902-05 | 411 | tic_up | next_lick | 20 | next | True | True | 0.43661 | 0.6083 | 0.61353 | 0.61095 | 0.53982 |
| A133-20230120-06 | 11 | tic_up | next_lick | 40 | next | True | True | 0.71542 | 0.76914 | 0.77596 | 0.78584 | 0.73285 |
| A213-20250916-04 | 167 | tic_up | next_lick | 60 | next | True | True | 0.86277 | 0.83922 | 0.87779 | 0.88845 | 0.93469 |
| A163-20230903-02 | 165 | tic_up | next_lick | 80 | next | True | True | 0.84005 | 0.92369 | 0.87047 | 0.88269 | 0.96604 |
| A214-20251009-04 | 277 | time_cell | last_lick | 20 | last | True | True | 0.59241 | 0.63753 | 0.41489 | 0.50099 | 0.46732 |
| A164-20230902-05 | 74 | time_cell | last_lick | 40 | last | True | True | 0.83247 | 0.74883 | 0.083282 | 0.78281 | 0.84898 |
| A132-20221124-02 | 112 | time_cell | last_lick | 60 | last | True | True | 0.9158 | 0.83814 | 0.32715 | 0.87254 | 0.92054 |
| A143-20230315-02 | 144 | time_cell | last_lick | 80 | last | True | True | 0.96 | 0.9013 | 0.22946 | 0.92011 | 0.93589 |

The plain model is retained in the tables but excluded from both figures. model_comparison_v2 uses primary revised anchoring against the group exponential (A) and Gaussian (B). fit_examples_v2 adds only the revised bump to the existing example curves and box plots. Canvas sizes are 13 by 5.8 inches and 13 by 15.5 inches, respectively, with fonts of at least 14 pt. Both PDFs were rendered and inspected, and pdffonts confirmed embedded CID TrueType subsets. Long bar p-labels were shifted inward to clear the plot borders; no analysis changed after inspecting the results.

The full-data failure is the non-primary next anchoring of a ramp-down cell, so every primary revised curve used for R2 and the example figure has fit_success=True. Both failed fits remain in the per-cell contrast and parameter records where applicable.
