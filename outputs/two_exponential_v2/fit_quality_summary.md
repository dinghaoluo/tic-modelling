# trial-averaged fit quality with the revised bump and plain exponential

The plain exponential remains in this table and is omitted from both figures; only the revised bump is added to the example curves and box plots. All curves use full-data parameters and the same observed trial-bin mask, alignment interpolation and minimum of 10 contributing trials as the original averaged-trace analysis. The original Gaussian, exponential and first-bump R2 values are unchanged, as are the 12 example IDs. The revised bump uses last anchoring for ramp-down TICs and time cells, and next anchoring for ramp-up TICs. Ramp-up traces run from negative time to zero relative to the next lick; both alignments remain in the table. These R2 values describe fitted averages, while the separate held-out contrasts test prediction.

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

Averaging and figure wall time: 30.8 s. Both figures use 13-inch-wide canvases and fonts of at least 14 pt; PDF fonts are embedded TrueType.
