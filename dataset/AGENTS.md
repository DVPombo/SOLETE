# AGENTS.md — orientation for AI coding agents

Read `README.md` first, then `docs/DATA_DICTIONARY.md`. This repo is the cleaning and
QC pipeline for the SOLETE dataset (wind + PV + meteorology, 1 s resolution, 39.48M rows,
2018-06-01 → 2019-08-31). The data files are not in the repo.

## Rules

- **Never modify the raw file** (`SOLETE_Pombo_1sec.h5`) and never overwrite an input.
  `clean_solete_1sec.py` and `export_parquet.py` refuse to write over their input; keep it that way.
- **Never commit data** (`*.h5`, `*.parquet`); they are git-ignored on purpose.
- **Do not load the 1-second file casually.** It is ~3 GB on disk, several GB in RAM.
  Read only the columns you need (`columns=[...]`), use chunks (`start`/`stop`) or the
  Parquet file with a time filter. Do not call `df.sort_index()` on the whole frame;
  `clean_solete_1sec.py` sorts column by column to avoid a MemoryError.
- **A changed rule means a changed document.** If you alter detection logic or thresholds,
  update `docs/CLEANING_DECISIONS.md` and `docs/QC_SCHEMA.md` in the same change.
- **Do not invent explanations.** Open questions are listed at the end of
  `docs/CLEANING_DECISIONS.md` (why `P_Gaia` is zero, sentinel completeness, ...). Say a
  cause is unknown rather than offering a plausible story, and report numbers from the run.
- There is no test suite. Verify changes by running on a small slice and reading the JSON
  report that every script prints (`solete_report.print_report`).

## Traps that produce wrong analyses

- `HUMIDITY[%]` is a **0–1 fraction**, not a percentage.
- Timestamps are **UTC**. `.h5` indexes are tz-naive; Parquet `timestamp` is tz-aware UTC.
- Resampled rows are labelled by interval **start**, `[T, T+period)`. The hourly file is `60min` (v3 and v4 name; `1h` is only an input alias in the code).
- `Pressure[mbar]` is valid on **2019-01-16 only**; elsewhere it is NaN.
- `P_Gaia[kW]` is confirmed real on **2018-08-31 and 2019-05-25 only**. Other zeros are unexplained: do not describe them as "turbine idle" or "no wind".
- `Azimuth[deg]`/`Elevation[deg]` are **computed** (pvlib), 0° = south, east negative, elevation negative at night. They are not measurements.
- Never average angles or flag codes arithmetically. `WIND_DIR[deg]` uses a circular mean; `<column>_qc` becomes `_qc_worst` + `_qc_frac_flagged` when resampled.
- QC codes are defined once, in `solete/qc_codes.py`. Code **6 (`QC_MODEL_SUBSTITUTED`) is platform-owned**:
  nothing in `dataset/` may emit it (`assert_pipeline_codes`), and the resampler drops every model-derived
  column so a 6 can never be averaged up. Model columns (`Pac`, `P_hybrid[kW]`, ...) are computed per
  resolution by `solete/expansion.py`, not here. A new rule takes the next free number after 11.
- File names are built in one place, `solete/paths.py` (`data_filename`, `release_path`): `SOLETE_Pombo_<res>_v4`, `SOLETE_Pombo_1sec_original_v4`. The cleaning rules are in `clean_solete_1sec.clean_block` and nowhere else; the build (`build_release.py`) only moves slices between the rule functions. Slices are cut only at `find_safe_cut` boundaries: a new run-based rule must be added to `unsafe_row_mask`/`safe_boundaries` or sliced cleaning will silently differ from whole-file cleaning (`tests/test_release_build.py` has the exactness test).
- For `P_Gaia` only, `_qc_frac_flagged` is always 1.0 (every row is code 7 or 10; checked by a test); use `_qc_worst`. Azimuth/Elevation have no flag column in v4 (D1); code 8 is reserved, not emitted by any v4 column.

## Using the data

Prefer the Parquet files. Filter on the flags rather than dropping values blindly:
`df[col + "_qc"] == 0` keeps untouched values; in resampled files filter on
`_qc_frac_flagged`. State which flags you excluded in anything you report.

## Layout

`pipeline/` code that produces the release · `docs/` decisions and schemas ·
`diagnostics/` read-only investigation scripts (not needed for normal use).
