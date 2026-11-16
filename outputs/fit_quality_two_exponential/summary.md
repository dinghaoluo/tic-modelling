# trial-averaged fit quality with the two-exponential bump

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
