# single-trial lick thresholds

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
