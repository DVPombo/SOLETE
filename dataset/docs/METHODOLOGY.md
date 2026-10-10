# Resampling methodology

How `pipeline/resample_solete.py` turns the cleaned 1-second file into the
1-minute, 5-minute and 60-minute files (`SOLETE_Pombo_{1min,5min,60min}_v4`; `1h` is only an input alias of `60min` in the code).
`pipeline/build_release.py` runs it slice by slice on whole days (a day boundary is a bucket boundary at every resolution), which gives the
same result as resampling the whole file; the empty buckets of a real gap stay NaN.

**Interval convention.** Left-closed, left-labelled: the bucket labelled `T`
covers `[T, T+period)`. The hourly row stamped `2019-01-01 03:00:00` is the
mean of the seconds from 03:00:00 to 03:59:59. Timestamps are UTC.

**Aggregation.**

| Columns | Aggregation | Why |
|---|---|---|
| `WIND_DIR[deg]` | circular mean (sin/cos vector average, `atan2`, wrapped to [0, 360)) | a plain mean of 359° and 1° gives 180° instead of ~0° |
| `Azimuth[deg]` (south-referenced, [-180, 180)) | circular mean, wrapped to [-180, 180) | the 1 s values jump from +180 to −180 at solar midnight; a plain mean of the bucket around it would give ~0 (south) |
| every other measurement column, including `Elevation[deg]` | arithmetic mean over the non-NaN seconds | NaNs left by cleaning are skipped, not treated as zero |
| `<column>_qc` flags (pipeline-owned, codes 0-5, 7-10) | **not averaged** — each becomes two columns, below | the mean of integer codes is meaningless |

| model-derived columns (`Pac`, `Pdc`, `TempModule`, `TempCell`, `P_Solar_clean[kW]`, `P_hybrid[kW]`, their flags, code 6) | **not resampled at all** — computed per resolution | see the next section |

**Flag columns in resampled files.** Each `<column>_qc` becomes:

- `<column>_qc_worst` — the highest-severity code present in the bucket
  (precedence in `QC_SCHEMA.md` §2).
- `<column>_qc_frac_flagged` — fraction of the bucket's seconds that were not
  `QC_OK` (0), so you can set your own tolerance (e.g. drop buckets > 10 % flagged).

A bucket with no source seconds at all (a genuine gap) gets NaN in both
columns, which distinguishes it from an all-OK bucket (`_worst == 0`,
`_frac_flagged == 0.0`).

**Caveats.**

- For `P_Gaia[kW]` no second is ever `QC_OK` (every row is code 7 or 10), so `P_Gaia[kW]_qc_frac_flagged` is a constant `1.0`. Use
  `_qc_worst` for it. (Azimuth/Elevation have no flag column at all in v4: decision D1.)
- `WIND_DIR[deg]_qc_frac_flagged` can reach ~0.78 in a one-minute bucket: wraps
  cluster in runs (mostly 2018-11-17), they are not isolated blips.
- When comparing with the originally published (v3) resampled files, check
  that file's interval labelling first; `diagnostics/compare_resolutions.py`
  does this before reporting any difference.

## Model columns are per resolution

The released full files follow one rule:

- **Measured columns and pipeline-owned flags** (codes 0–5, 7–10) are resampled from the cleaned 1-second data, as above.
- **Model-derived columns** (`Pac`, `Pdc`, `TempModule`, `TempCell`, `P_Solar_clean[kW]`, `P_hybrid[kW]` and
  their flags, including code 6) are computed at each resolution by the same function, `solete.expansion.expand_physical`, from that
  resolution's own cleaned inputs. They are **never averaged up** from a finer resolution, and code 6 is never carried upward.
  `resample_solete.py` enforces this: it drops `solete.qc_codes.MODEL_DERIVED_COLUMNS` even if the 1-second input contains them, and replaces any
  non-pipeline code inside a pipeline flag column by `QC_OK` before aggregating.

**Why.** The PV model and the substitution rule are not scale-free. The cell-temperature and power terms are nonlinear, power is clipped at zero
and at the inverter maximum, and `Pac >= 1.5 * P_Solar` is a threshold, so it flags different instants at 1 s than on hourly means.
The hourly `Pac` is therefore *not* the mean of the 1-second `Pac`, and the hourly substitution flag is not the share of flagged seconds.
At 1 s the model columns are instantaneous model estimates, not physically validated (module thermal inertia is minutes), and the 1 s
substitution flag is not comparable with the hourly one. This is an intended design, not a defect.

**Size of the effect.** Difference between `expand(resampled inputs)` (what the files contain) and `resample(expand(1 s inputs))` (the average of the
1-second model output), from `scripts/expansion_checks.py effect --days 14`. **These numbers are from SYNTHETIC 1-second data**
(`solete/synthetic.py`; the real 1-second file was not available when this was written) and describe that generator, not the dataset.
Re-run the script with `--input` on the real cleaned 1-second file to get the real table.

**Model columns:** expand(resampled inputs) vs resample(expand(1 s inputs))

| resolution | column | mean of resampled 1 s model | mean of model at resolution | mean diff | mean abs diff | rel. mean abs diff % | max abs diff |
|---|---|---|---|---|---|---|---|
| 1min | `Pac` | 2.0388 | 2.0404 | +0.0015 | 0.0015 | 0.08 | 0.060 |
| 1min | `Pdc` | 2.0783 | 2.0799 | +0.0016 | 0.0016 | 0.08 | 0.061 |
| 1min | `TempModule` | 20.3665 | 20.3616 | -0.0049 | 0.0103 | 0.05 | 0.227 |
| 1min | `TempCell` | 21.2498 | 21.2449 | -0.0049 | 0.0103 | 0.05 | 0.227 |
| 1min | `P_Solar_clean[kW]` | 1.8914 | 1.8911 | -0.0004 | 0.0012 | 0.06 | 1.378 |
| 1min | `P_hybrid[kW]` | 1.9325 | 1.9321 | -0.0004 | 0.0012 | 0.06 | 1.378 |
| 5min | `Pac` | 2.0388 | 2.0416 | +0.0028 | 0.0028 | 0.14 | 0.031 |
| 5min | `Pdc` | 2.0783 | 2.0811 | +0.0028 | 0.0028 | 0.14 | 0.032 |
| 5min | `TempModule` | 20.3665 | 20.3593 | -0.0072 | 0.0105 | 0.05 | 0.119 |
| 5min | `TempCell` | 21.2498 | 21.2426 | -0.0072 | 0.0105 | 0.05 | 0.119 |
| 5min | `P_Solar_clean[kW]` | 1.8914 | 1.8893 | -0.0021 | 0.0050 | 0.27 | 1.948 |
| 5min | `P_hybrid[kW]` | 1.9325 | 1.9304 | -0.0021 | 0.0050 | 0.26 | 1.948 |
| 60min | `Pac` | 2.0388 | 2.0535 | +0.0147 | 0.0147 | 0.72 | 0.162 |
| 60min | `Pdc` | 2.0783 | 2.0933 | +0.0150 | 0.0150 | 0.72 | 0.165 |
| 60min | `TempModule` | 20.3665 | 20.3448 | -0.0216 | 0.0620 | 0.30 | 0.637 |
| 60min | `TempCell` | 21.2498 | 21.2281 | -0.0216 | 0.0620 | 0.29 | 0.637 |
| 60min | `P_Solar_clean[kW]` | 1.8914 | 1.8575 | -0.0340 | 0.0530 | 2.80 | 1.505 |
| 60min | `P_hybrid[kW]` | 1.9325 | 1.8985 | -0.0340 | 0.0530 | 2.75 | 1.505 |

**Substitution flag (code 6: `Pac >= 1.5 * P_Solar` and `Pac > 0`)**

| resolution | buckets | flagged at 1 s (share of 1 s rows) | flagged at resolution | disagrees w/ majority-of-1 s | disagrees w/ any-1 s |
|---|---|---|---|---|---|
| 1min | 20,160 | 5.36% | 5.46% | 0.05% | 1.05% |
| 5min | 4,032 | 5.36% | 5.63% | 0.20% | 2.85% |
| 60min | 336 | 5.36% | 6.25% | 2.38% | 19.94% |

Reading the synthetic table: power columns differ by well under 1 % at 1 and 5 minutes and by about 0.7 % (`Pac`) to 2.8 % (`P_Solar_clean[kW]`,
because substitution enters) at one hour; temperatures by 0.01 °C (1–5 min) to 0.06 °C (1 h) on average. The share of rows flagged is similar at every
scale (5.4 % at 1 s, 5.5–6.3 % at coarser steps), but *which* rows are flagged is not: the hourly flag disagrees with the majority of its seconds in about 2 %
of hours and with "any second flagged" in about 20 %. Do not read an hourly code 6 as "x % of this hour's seconds were substituted".
