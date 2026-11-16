# latency, relaxation and peak width

Both parameters must be away from either saved bound; exclusion uses the wider-window bound flags, with no refitting. Spearman correlations are computed across retained cells and separately within each session; raw two-sided Wilcoxon p-values test the defined session correlations against zero. Rise mu retains its saved time-before-next-lick definition. IQRs describe variation between sessions, not uncertainty in a pooled coefficient.

| group | x | y | n | excluded | pooled_r | sessions_defined | median_session_r | session_r_q25 | session_r_q75 | wilcoxon_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tic_down | decay_mu | decay_tau | 207 | 197 | -0.28875 | 15 | -0.25275 | -0.48571 | -0.079798 | 0.012451 |
| tic_up | rise_mu | rise_tau | 282 | 197 | -0.57879 | 15 | -0.6 | -0.67404 | -0.44761 | 0.00042725 |
| time_cell | decay_mu | decay_tau | 735 | 442 | -0.17754 | 15 | -0.35743 | -0.41282 | -0.031962 | 0.0083618 |
| time_cell | gaussian_mu | gaussian_sigma | 1080 | 97 | -0.042196 | 15 | -0.046931 | -0.21301 | 0.08468 | 0.20776 |

Wall time: 2.4 s. Excluding bounded cells can alter the sampled relationship; the pooled and session estimates are reported together so a heavily represented session can be recognised.
