# shared-reference peak location

The fitting window runs from 1 s before the last lick to the first lick of the next trial. Ramp-down TICs use decay_mu and Time cells use gaussian_mu, both referenced to the last lick. Ramp-up TICs use the session median observed trial interval T_k minus rise_mu.

                       group   median  p for median                 comparison                                                   convention
               Ramp-down TIC 0.341514      0.719727                  Time cell                              decay_mu, relative to last lick
                 Ramp-up TIC 4.083549      0.000061                  Time cell                           -rise_mu, relative to interval end
                  Time cells 0.461805           NaN                  reference                                                          NaN
Ramp-down TIC vs Ramp-up TIC      NaN      0.000122 shared last-lick reference ramp-down decay_mu; ramp-up session median T_k minus rise_mu

Session-level medians and the three pairwise tests are in `peak_location_shared_reference_session_medians.csv`. All tests are two-sided Wilcoxon signed-rank tests on session medians, n = 15 sessions.
