# progress

## 4 november 2026

I set up the repository around the fifteen 4-s delay sessions, the source paths, the binned data and the receptive-field models. The recording folders remain read only. The dedicated Python 3.11 environment lives under data/tic-modelling.

The first source check found that one of the valid-trial files was absent. I recomputed the valid rows from trialsCue using the specified first- and last-lick rule, checked the available comparison files, and built the trial-binned data from those rows.

## 5 november 2026

The first complete fitting run finished with Gaussian, exponential decay, exponential rise and polynomial guard shapes. I saved the trial folds, out-of-fold predictions, full-data parameters, fit flags and summary figures. Failed optimisation stages remain marked in the diagnostics.

I then rebuilt the dataset from the paired FirstLick interval, keeping the source pairing and the existing bin masking. The revised run saved out-of-fold predictions for every observed trial bin and used the paired trial-error score for model preferences. The earlier output and figure directories were protected throughout.

The saved single-trial comparison now supplies the main shape test. It uses the held-out trial errors, a shared flat residual variance within each cell, trial resampling and session-level tests. The parameter table includes peak position, width or time constant, bounds and session comparisons.

The wider-window, recovery, label-stability, epoch and peak-position checks were run in their own output directories. I kept the original model outputs unchanged and left failed optimisation stages visible.

## 6 november 2026

I checked the requested window beyond the first lick before starting another fit. Some source arrays end before the full extension, and the synthetic TIC-selection check would require a rule for constructing activity outside the saved recovery window. I recorded the audit and stopped before fitting that follow-up.

I split the ramp-up TIC traces into the existing early- and late-lick groups and plotted the population and session means. Both groups rise gradually, while the session traces vary substantially. The split does not give a clean exponential rise, so ramp-up remains a group-level unresolved shape question.

I checked the saved threshold widths against the source TIC and time-cell order. The rule uses a low-pass trace, the mode as baseline, and baseline plus 0.5 SD as the crossing threshold. Widths that reach the recording limit are marked as censored. The width table and figure are saved under outputs/width_summary and figures/width_summary.

The next step is an evidence audit of the formal comparison, including the unresolved and flat-cell proportions, before I draft any response-letter or reviewer-facing figure claims.
