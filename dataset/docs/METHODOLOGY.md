# Resampling methodology

How `pipeline/resample_solete.py` turns the cleaned 1-second file into the
1-minute, 5-minute and 1-hour files. (`1h` is the same resolution the
original release called `60min`.)

**Interval convention.** Left-closed, left-labelled: the bucket labelled `T`
covers `[T, T+period)`. The hourly row stamped `2019-01-01 03:00:00` is the
mean of the seconds from 03:00:00 to 03:59:59. Timestamps are UTC.

**Aggregation.**

| Columns | Aggregation | Why |
|---|---|---|
| `WIND_DIR[deg]` | circular mean (sin/cos vector average, `atan2`, wrapped to [0, 360)) | a plain mean of 359° and 1° gives 180° instead of ~0° |
| every other measurement column, including `Azimuth[deg]` and `Elevation[deg]` | arithmetic mean over the non-NaN seconds | NaNs left by cleaning are skipped, not treated as zero |
| `<column>_qc` flags | **not averaged** — each becomes two columns, below | the mean of integer codes is meaningless |

Only measured/pipeline columns are resampled. The deterministic model columns
`Pac`, `Pdc`, `TempModule`, `TempCell`, `P_Solar_model_substituted`,
`P_Solar_clean[kW]`, `P_hybrid[kW]` and their QC fields are excluded. They are
recomputed by `solete.expansion.expand_physical` from each target resolution's
own cleaned inputs. Code 6 is evaluated at each resolution and is never
propagated from a finer file.

`TempModule_RP` is not a release expansion column. Its thermodynamic model
carries module temperature between adjacent rows and applies a sequential
gradient limiter, so it is neither row-wise nor chunk-invariant. It remains
an opt-in ML feature in `ExpandSOLETE`.

**Flag columns in resampled files.** Each `<column>_qc` becomes:

- `<column>_qc_worst` — the highest-severity code present in the bucket
  (precedence in `QC_SCHEMA.md` §2).
- `<column>_qc_frac_flagged` — fraction of the bucket's seconds that were not
  `QC_OK` (0), so you can set your own tolerance (e.g. drop buckets > 10 % flagged).

A bucket with no source seconds at all (a genuine gap) gets NaN in both
columns, which distinguishes it from an all-OK bucket (`_worst == 0`,
`_frac_flagged == 0.0`).

**Caveats.**

- For `P_Gaia[kW]`, `Azimuth[deg]` and `Elevation[deg]` no second is ever
  `QC_OK`, so `_qc_frac_flagged` is a constant `1.0`. Use `_qc_worst` for those.
- `WIND_DIR[deg]_qc_frac_flagged` can reach ~0.78 in a one-minute bucket: wraps
  cluster in runs (mostly 2018-11-17), they are not isolated blips.
- When comparing with the originally published (v3) resampled files, check
  that file's interval labelling first; `diagnostics/compare_resolutions.py`
  does this before reporting any difference.

## Why model columns are recomputed

King's PV model is nonlinear in irradiance, cell temperature and wind speed,
clips power, and feeds a threshold decision (`Pac >= 1.5 * P_Solar[kW]`). It
does not commute with averaging. The following comparison used the repository's
actual Python `expand_physical` function on six deterministic synthetic hours
(21,600 one-second rows). Values are mean absolute differences between
`expand(mean(inputs))` and `mean(expand(inputs))`; power is kW and temperature
is °C.

| Resolution | `Pac` | `Pdc` | `TempModule` | `TempCell` | `P_Solar_clean[kW]` | `P_hybrid[kW]` |
|---|---:|---:|---:|---:|---:|---:|
| 1 min | 0.015394 | 0.015692 | 0.040314 | 0.040314 | 0.479772 | 0.479772 |
| 5 min | 0.033983 | 0.034641 | 0.064449 | 0.064449 | 0.492192 | 0.492192 |
| 60 min | 0.072363 | 0.073765 | 0.058724 | 0.058724 | 0.478257 | 0.478257 |

For the boolean/model-QC columns, averaging the 1-second flag gives the
fraction of seconds substituted, not a meaningful target-resolution flag.
The comparison below shows its absolute difference from the independently
computed target flag and two explicit disagreement definitions. The
`P_Solar[kW]_qc` and `P_hybrid[kW]_qc` majority disagreement rates are the
same as the boolean; `_qc_source` changes on those same buckets in this
synthetic case because wind QC is all zero.

| Resolution | Substitution-fraction MAE | Disagrees with 1 s majority | Disagrees with any 1 s flag |
|---|---:|---:|---:|
| 1 min | 0.408426 | 15.000% | 22.500% |
| 5 min | 0.429815 | 1.389% | 9.722% |
| 60 min | 0.432870 | 0.000% | 0.000% |

These numbers characterize this synthetic signal, not the released data.
They demonstrate why the release rule is intentional. At 1 second the model
columns are instantaneous estimates; they are not physically validated
module-temperature observations because real module thermal inertia is on
the order of minutes. A 1-second substitution flag must not be interpreted as
equivalent to an hourly one.

## Bounded-memory execution

`solete.expansion.iter_hdf_slices` reads both pandas fixed-format v3 files
and table-format pipeline files in bounded row slices.
`iter_expanded_hdf_slices` applies `expand_physical` independently to each
slice without modifying or loading the complete input. On the supplied real
`SOLETE_Pombo_1sec.h5`, Python processed all 39,484,801 rows in 15 slices of
at most 2,678,400 rows. The run took 42.24 seconds and reached 1,712.9 MiB
peak RSS on Windows, below the 3 GiB target. It produced finite `Pac` for all
rows and marked 16,758,568 rows as model-substituted. These are execution
validation counts, not new data-quality findings.
