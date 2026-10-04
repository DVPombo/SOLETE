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
