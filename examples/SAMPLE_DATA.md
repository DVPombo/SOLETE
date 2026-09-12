# SAMPLE_DATA.md — Notebook sample files

Two new, small `.h5` files at the repo root, both cut from the real
`SOLETE_Pombo_60min.h5` record (never synthetic). They exist to lower the barrier to
entry for `examples/*.ipynb` — small enough to be a fast, lightweight quickstart, but
chosen specifically to contain real examples of the QC issues documented in
`QC_SCHEMA.md`/`KNOWN_ISSUES.md`, which `SOLETE_short.h5` (see below) does not.

Both files are produced by slicing `SOLETE_Pombo_60min.h5` **after** sorting its index
(`KNOWN_ISSUES.md` finding #8: the file's rows are not stored in chronological order on
disk). Both keep the raw, as-delivered columns only — the QC `_qc` columns and PV-model
expansion columns (`Pac`, `Pdc`, `P_Solar_model_substituted`, etc.) are added at load time
by `Functions.py::import_SOLETE_sample()` (a new convenience wrapper, see below),
the same way `import_SOLETE_data()` adds them for the full-size files — not baked into the
saved file.

## `SOLETE_sample.h5` — main Phase 3 sample (Tasks 3.2, 3.3, 3.4)

- **Date range:** 2018-12-01 00:00 → 2019-02-28 23:00 (90 days, 2,160 rows, hourly).
- **Why this window:** All three candidate windows checked (see below) comfortably
  covered every QC issue type, since `Pressure[mbar]`'s sentinel problem and the
  `HUMIDITY[%]`/`WIND_DIR[deg]` out-of-range rows are spread across essentially every
  month of the 15-month record, and the one populated azimuth/elevation day
  (2019-01-16, `KNOWN_ISSUES.md` finding #7) falls naturally inside any window spanning
  January 2019. Given that, the deciding factor was the notebook use case: Dec–Feb gives
  the forecasting notebooks (Tasks 3.4/3.5) the most history of the options considered,
  while staying small enough to load and run in well under a minute.
- **Issue-type coverage, verified by running the sample through
  `apply_qc_flags`/`build_raw_value_qc_rules` and the substitution-flag logic in
  `ExpandSOLETE()`** (exact counts, out of 2,160 rows):

  | Flag type | Column | Rows flagged |
  |---|---|---|
  | `physically_implausible` (pressure sentinels) | `Pressure[mbar]_qc` | 2,147 |
  | `physically_implausible` (humidity > 1) | `HUMIDITY[%]_qc` | 42 |
  | `physically_implausible` (wind dir ≥ 360°) | `WIND_DIR[deg]_qc` | 31 |
  | `missing` (azimuth ≈ 0) | `Azimuth[deg]_qc` | 2,150 (10 real values, on 2019-01-16) |
  | `missing` (elevation ≈ 0) | `Elevation[deg]_qc` | 2,151 (9 real values, on 2019-01-16) |
  | `suspected_curtailment_or_model_substituted` | `P_Solar[kW]_qc` | 1,272 |

  Every flag type is non-zero, including the two that `SOLETE_short.h5` cannot
  demonstrate at all (see below).

## `SOLETE_sample_wind.h5` — dedicated wind-forecasting sample (Task 3.5 only)

- **Date range:** 2019-04-26 00:00 → 2019-06-24 23:00 (60 days, 1,440 rows, hourly).
- **Why a second, separate sample:** while building the wind-forecasting notebook on the
  main sample above, `P_Gaia[kW]` (wind turbine active power) turned out to be exactly
  zero for the *entire* Dec–Feb window — and in fact for almost the entire 15-month
  source file. Checking the full `SOLETE_Pombo_60min.h5`: only 48 of 10,969 rows (0.44%)
  have `P_Gaia[kW] > 0`, and they fall on exactly two isolated calendar days —
  2018-08-31 and 2019-05-25 — with every other hour reading exactly zero, including
  thousands of hours with wind speed well above the turbine's 3.5 m/s cut-in. This is a
  real, previously-unnoticed characteristic of the delivered data (most plausibly the
  turbine was out of service for most of the 15-month record — Gaia has a documented
  history of mechanical problems — though that hasn't been independently confirmed).
  A persistence baseline on an all-zero target "forecasts" it perfectly (0.000 MAE/RMSE),
  which teaches a newcomer nothing, so a plain reuse of the main sample would make Task
  3.5's notebook look broken rather than illustrate anything real.
  **Per the maintainer, this finding is not being added to `KNOWN_ISSUES.md` or
  `DATA_DICTIONARY.md`** — it's flagged here and in `examples/04_wind_forecasting.ipynb`
  itself instead, since it's specific to why that one notebook uses a different sample.
- **Resolution:** a second, dedicated sample centered on the 2019-05-25 active day,
  with 60 days of surrounding context so the notebook still has enough rows for a
  train/test split and lagged features, even though `P_Gaia[kW]` remains zero for all
  but 24 of the 1,440 rows.
- **Issue-type coverage** (same method as above, out of 1,440 rows): `Pressure[mbar]_qc`
  1,440 flagged, `HUMIDITY[%]_qc` 19, `WIND_DIR[deg]_qc` 11, `Azimuth[deg]_qc`/
  `Elevation[deg]_qc` unpopulated (2019-01-16 falls outside this window — expected and
  fine, since this sample's only job is the wind-forecasting notebook, not full QC
  coverage), `P_Solar[kW]_qc` 358 substituted. `P_Gaia[kW] > 0`: 24 rows (exactly the
  2019-05-25 day).

## `SOLETE_short.h5` — kept as-is, unchanged

`SOLETE_short.h5` (24 rows, one day, hourly) is still referenced directly by
`tests/test_qc_flags.py` and `RunMe.py`, so it is **kept exactly as it was** — not
modified, retired, or replaced by the new samples above. It remains the fastest possible
smoke-test file; it just isn't representative enough (zero QC-flagged rows for pressure,
humidity, wind direction, or azimuth/elevation — see `availability_report.csv`) to be the
basis for the data-quality notebook, which is why Task 3.1 exists.

## Loading these samples

Both files load through the real `Functions.py` entry point, not a bypass — a new small
wrapper, `import_SOLETE_sample(path, Control_Var, PVinfo, WTinfo)`, mirrors
`import_SOLETE_data()`'s `'Build'` branch (raw-value QC flags, then `ExpandSOLETE()`) for
an explicit file path, since `import_SOLETE_data()` itself only accepts the four fixed
`resolution` values and derives `SOLETE_Pombo_<resolution>.h5` as the filename — it has no
way to point at an arbitrary sample file without a naming collision. This wrapper is
additive; `import_SOLETE_data()`'s own behavior for the real files is unchanged. See
`examples/01_dataset_overview.ipynb` for the pattern in use.
