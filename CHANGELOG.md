# Changelog

All notable changes to the SOLETE platform are documented in this file.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Entries for v1.0 through v3.0 were backfilled from the git history and condensed from [releases/](releases/); see those files for more detail, and see `releases/v3.0_notes.md` for the full corrigendum text.

## [Unreleased]
### Fixed
- **Phase 0.5** (commit `5eb11d0`) — Three fixes, backfilled here per `KNOWN_ISSUES.md`'s "Already-fixed issues" section and housekeeping gap #9 (see `git show 5eb11d0` for the diff):
  - `PV_Performance_Model()`'s inverter-capacity clamp used whole-DataFrame boolean-mask assignment (`Results[mask] = value`), which applies the scalar to *every* column on the masked rows — silently clobbering `Tm`, `Tc`, `Pmp_panel`, `Pmp_array`, and `eff_inv` alongside the intended `Pac_<pv>` column. Fixed to `Results.loc[mask, 'Pac_' + pv] = value`, confined to the intended column.
  - `Rincon_Pombo_ThermodynamicModel()`'s per-timestep loop indexed pandas Series with integer positions (e.g. `data['HUMIDITY[%]'][i]`), relying on pandas' old label→positional fallback for non-integer (DatetimeIndex) indexes. Modern pandas (3.0+) removed that fallback and raises `KeyError` instead. Fixed by operating on plain `numpy` arrays (`.to_numpy()`) so integer indexing is unambiguous.
  - `ExpandSOLETE()` silently substitutes `P_Solar[kW]` with the modeled `Pac` on rows where `Pac >= 1.5 * P_Solar[kW]` (noise/curtailment cleaning), with no way downstream to tell which rows had been substituted. Added a new boolean column, `P_Solar_model_substituted`, alongside the substitution so it's traceable — documented in `DATA_DICTIONARY.md`.

### Added
- New QC flag layer (`QC_SCHEMA.md`, `Functions.py::apply_qc_flags` + `build_raw_value_qc_rules`/`build_substitution_qc_rule`, `tests/test_qc_flags.py`) turning the six findings in `KNOWN_ISSUES.md` into explicit, queryable `<column>_qc` columns instead of prose. Flags are mutually exclusive (one value per cell, see `QC_SCHEMA.md` for the bitmask-vs-exclusive reasoning). Verified against both real files (`SOLETE_short.h5`, `SOLETE_Pombo_60min.h5`):
  - `Pressure[mbar]_qc`: flags known sentinels (`1000.0`/`2000.0`/`3000.0`) plus a general out-of-range safety net (<870 or >1085 mbar). 60min file: 10,477 rows @ 1000.0 (95.51%), 477 @ 2000.0 (4.35%), 2 @ 3000.0 (0.018%) — 10,956 rows flagged total, matching `KNOWN_ISSUES.md` exactly. Short file: 0 flagged.
  - `HUMIDITY[%]_qc`: flags value > 1.0 or < 0.0. 60min file: 188 rows (1.71%). Short file: 0.
  - `WIND_DIR[deg]_qc`: flags value >= 360.0 or < 0.0. 60min file: 103 rows (0.94%). Short file: 0.
  - `Azimuth[deg]_qc` / `Elevation[deg]_qc`: flags value == 0.0 as `missing` (these columns are a later, non-DTU-release addition — see `QC_SCHEMA.md` §6). 60min file: 10,959/10,960 rows (99.91%/99.92%) — matches `KNOWN_ISSUES.md`'s "99.9%" finding. Not present in the short file.
  - `P_Solar[kW]_qc`: folds the existing `P_Solar_model_substituted` boolean (Phase 0.5) into the same convention. 60min file: 4,204 rows (38.33%). Short file: 5 rows (20.83%). Matches `KNOWN_ISSUES.md`'s ~38%/~21% exactly.
  - Raw-value checks run in `import_SOLETE_data()` (both `Build` and `Import` branches, directly on the raw columns — self-healing against the pre-existing `PossibleFeatures` drop gap, see below); the substitution mapping runs in `ExpandSOLETE()` right after `Pac` is computed.
- New `scripts/availability_report.py`, extending `scripts/inspect_dataset.py`'s HDF5-key discovery, computing per-column-per-file completeness (expected vs. actual sample count, read from the file's actual time span and inferred spacing rather than a hardcoded resolution) and QC-flag breakdown. Output: `availability_report.csv`.

### Known issues (surfaced while building the QC layer, not fixed here)
- `P_Solar_model_substituted` (and now the new `_qc` columns) are only protected from being dropped on a save→`Import` round trip if the caller's own `Control_Var['PossibleFeatures']` literal lists them — the example list in `MLForecasting.py` doesn't. `ExpandSOLETE`'s `list_expansion` tracking only feeds a diagnostic print, not the actual Import-time drop logic. Documented in `QC_SCHEMA.md` §7; not changed here since fixing the drop logic itself is a bigger behavior change than this phase's scope (adding flags, not fixing the registration mechanism).
- Row order in `SOLETE_Pombo_60min.h5` is not chronological on disk (confirmed independently via raw `h5py` read of the `DATA/axis1` index array, not a pandas artifact — file MD5 `c0795d19ea933fec892271d90f6cedb4`). Values are correct once sorted; this is a file-level issue, not a per-cell one, so it doesn't fit the `<column>_qc` pattern. Per maintainer decision (2026-09-10), left as a documented caveat only this phase — revisit after the current GitHub pass, likely alongside how the source resolution files get concatenated/resampled into the 60min file.
- `Azimuth[deg]`/`Elevation[deg]` are a later, non-DTU-release addition (maintainer-added via GPS + timestamp through an external Python library, not present in the original SOLETE paper's variable list). The one populated day (2019-01-16) checks out against an independent `pvlib` solar-position calculation — elevation within ~0.1-1° at midday, azimuth consistent to ~11-12° (south-referenced convention) for the 8 rows well above the horizon — so the populated values look genuine, just on a possibly different time/azimuth basis. The real issue is coverage (99.9% unpopulated), not correctness. Recomputing/backfilling is out of scope for this phase; revisit alongside the row-order issue.

### Changed
- Bumped all pinned dependencies in `requirements.txt` to their latest stable releases (pandas 3.0.5, numpy 2.5.3, matplotlib 3.11.1, scikit-learn 1.9.0, TensorFlow 2.21.0 / Keras 3.15.1, CoolProp 8.0.0) and the target interpreter to Python 3.13, replacing the original Python 3.9.12 / pandas 1.5.0 / TensorFlow-Keras 2.10.0 / scikit-learn 1.1.2 pins, which no longer install on a current Python.
- Included `requirements.txt`to ease installation.
- Updated `Dockerfile` base image from `python:3.9-slim` to `python:3.13-slim` to match.

### Known issues (not fixed here — packaging only)
- `Functions.py`'s `post_process()` called `sklearn.metrics.mean_squared_error(..., squared=False/True)`, an argument scikit-learn removed in 1.4+. Replaced with `root_mean_squared_error()` (for the former `squared=False` calls) and bare `mean_squared_error()` (for the former `squared=True` calls, now equivalent to the old default). This unblocks `MLForecasting.py`'s error-computation step under the new pins.
- `Functions.py`'s `post_process()` calls `sklearn.metrics.mean_squared_error(..., squared=...)`, an argument scikit-learn has since removed. This breaks `MLForecasting.py`'s error-computation step under the new pins. 
- While verifying the above against the real pinned stack, `post_process()` raised a second, unrelated `KeyError` from `rmse.mean()[0]` / `mae.mean()[0]` / `mse.mean()[0]`: pandas 3.0 removed the integer-position fallback in `Series.__getitem__`, so indexing a label-indexed Series with `[0]` now raises instead of warning. Changed to `.mean().iloc[0]` (explicit positional access) in all three spots.
- Verified: imported `Functions.py` directly and called `post_process()` against the real pinned stack (pandas 3.0.5, numpy 2.4.4, scikit-learn 1.9.0, TensorFlow 2.21.0, CoolProp 8.0.0) with synthetic Observed/Forecasted/Persistence data — ran end-to-end with correct RMSE/MSE/MAE output and no errors. Also re-ran `RunMe.py` end-to-end after the change to confirm no regression. Both were run on Python 3.12 (no 3.13 interpreter available in the sandbox); wheel availability for 3.13 was already confirmed separately for every pinned package.
- Not yet run: the real `MLForecasting.py` training pipeline against actual data (that needs a full LSTM/RF/SVR training run with real datasets, which wasn't exercised here) — the synthetic-data test above targets the specific bug in `post_process()`, not a full pipeline regression test.

### Fixed
- **Task 5.0** — Fixed a sample-file path mismatch in all four `examples/*.ipynb` notebooks: each called `import_SOLETE_sample('../SOLETE_sample.h5', ...)` (or `'../SOLETE_sample_wind.h5'` for `04_wind_forecasting.ipynb`), i.e. one directory above the notebook, but the sample files ship *inside* `examples/` alongside the notebooks themselves. Since the normal Jupyter/Colab default working directory is the notebook's own directory, this made every notebook fail with a file-not-found error on a clean checkout unless the user happened to `cd` up first. Fixed by changing the load-call path strings from `'../SOLETE_sample.h5'` / `'../SOLETE_sample_wind.h5'` to `'SOLETE_sample.h5'` / `'SOLETE_sample_wind.h5'` (relative to the notebook's own directory, matching where the files actually are) — chosen over moving the `.h5` files to the repo root because it's a one-line string fix with no git-history renames, and it keeps `examples/SAMPLE_DATA.md`'s existing description of the files' location accurate as-is. Also corrected the matching Colab-bootstrap cell's fallback-download destination (`dest = "../" + SAMPLE_FILE` → `dest = SAMPLE_FILE`) in all four notebooks so a from-scratch Colab run would download to the same corrected location, not the old mismatched one. Verified by running `jupyter nbconvert --execute` on all four notebooks from their own directory against a clean checkout — all four now execute end to end with zero error cells.

### Added (Phase 6 — hybrid wind+solar forecasting)
- New derived `P_hybrid[kW]` column (`= P_Solar[kW] + P_Gaia[kW]`, computed in `ExpandSOLETE()` after substitution/zero-smoothing) plus its own QC columns, `P_hybrid[kW]_qc` and `P_hybrid[kW]_qc_source` (`QC_SCHEMA.md` §8): the hybrid inherits whichever constituent's QC flag is higher-precedence, with `_qc_source` naming which constituent (or `'none'`) produced it. No wind (`P_Gaia[kW]`) QC rule exists yet, so today this is equivalent to `P_Solar[kW]_qc` — pinned by the new regression tests added in Phase 7 Session 1 (`tests/test_qc_flags.py`).
- New `task6_2_hybrid_joint_vs_independent.py` (joint-vs-independent AR(p) forecasting comparison for `P_hybrid[kW]`, `results/hybrid_joint_vs_independent.json`) and `task6_3_ramp_rate_analysis.py` (ramp-rate distribution comparison plus a two-day case study, `results/hybrid_ramp_rate_summary.json` and accompanying figures). Per Task 6.0's decision (path (b), see `KNOWN_ISSUES.md` #10), both are scoped and reported as infrastructure/methodology deliverables on wind-degenerate data, not as a "joint beats independent" or "wind smooths ramps" finding — see each script's module docstring and `BENCHMARKS.md`'s hybrid section for the full caveats.
- New `examples/05_hybrid_forecasting.ipynb`, walking through the hybrid column, its QC inheritance, and both task scripts above.
- New `BENCHMARKS.md` section for `P_hybrid[kW]`, and `KNOWN_ISSUES.md` #10 documenting `P_Gaia[kW]`'s near-total zero-degeneracy (99.56% exactly zero across the full record; only 0.81% of the test split's rows are wind-active) with two unconfirmed candidate explanations (turbine downtime vs. an aggregation-pipeline artifact) — root cause not established in this repo.

<!-- add further entries here as work lands -->

## [3.0] - 2023-07-20
### Fixed
- Train/validation/test data-splitting bug (**corrigendum**: affected all results computed with v2.3 or earlier).
- RMSE calculation bug in results postprocessing (**corrigendum**: affected all results computed with v2.3 or earlier).
- Error computation and postprocessing improved across all ML models as part of the same fix.

### Changed
- Data preprocessing now adapts to different combinations of previous-samples and forecast horizons, for both ensemble and ANN methods.
- Memory handling improved; code simplified.
- Documentation enhanced.

> **Correction from v2.3:** the train/val/test split and the RMSE calculation both contained bugs, present through v2.3 and fixed here in v3.0. Results computed with v2.3 or earlier are not directly comparable to v3.0+. See `releases/v3.0_notes.md`.

## [2.3] - 2023-07-08
### Fixed
- Execution no longer stops when using the 1-second and 1-minute resolution versions of the dataset.

## [2.2] - 2023-06-30
### Fixed
- `RunMe_matlab.m` now re-sorts the SOLETE file by timestamp on import, correcting row reordering that some MATLAB versions introduced, and making the timestamp column easier to convert to MATLAB's `DateTime` type.

## [2.1] - 2023-04-27
### Added
- `RunMe_matlab.m`, a MATLAB script to import the SOLETE dataset directly without going through Python.

## [2.0] - 2023-04-11
### Added
- Ability to further expand the dataset, save the expanded version, and re-import it later.
- A function that catches common user errors and reports them with a clearer message.
- `CITATION.cff` for GitHub's "Cite this repository" feature.

### Changed
- Documentation and comments improved throughout in preparation for this release.

## [1.0] - 2022-02-02
### Added
- First tagged, citable release: minimum running scripts to load the SOLETE dataset and reproduce the "Data in Brief" paper's review materials.
- Initial `README.md` describing the dataset and repository purpose.
