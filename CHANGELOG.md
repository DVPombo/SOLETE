# Changelog

All notable changes to the SOLETE platform are documented in this file. From latest to oldest as you scroll down.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Entries for v1.0 through v3.0 were backfilled from the git history; see the "Corrigendum" section of [README.md](README.md) for the full corrigendum text.

## [Unreleased]
This section tracks forecasting-platform work on top of the already-released v4 **dataset** (`## [4.0]`
below is the dataset release; the platform's own version, `solete.__version__`, is still `4.0.0.dev0`).

### Added — CLI/config file for `scripts/quickstart/MLForecasting.py`
- `MLForecasting.py` no longer requires editing Python source to configure a run. Its `Control_Var`
  dict (the tier-1 scalars plus the five per-model hyperparameter dicts: RF/SVM/LSTM/CNN/CNN_LSTM) is
  now built by new `scripts/quickstart/ml_config.py` from, in increasing precedence: the shipped
  `scripts/quickstart/configs/default_control_var.json` < an optional `--config path/to/file.json`
  < explicit CLI flags (`--resolution`, `--mltype`, `-H/--horizon`, `--pre/--lookback`,
  `--train-val-test`, `--scaler`, `--train/--import-model`, `--save-model/--no-save-model`,
  `--random-seed`, `--build/--import-solete`, `--use-val-as-train/--no-use-val-as-train`; run
  `python MLForecasting.py -h` for the full list). Field meanings (including the ones that used to
  be inline comments next to the old dict literal) are documented in
  `scripts/quickstart/configs/README.md`.
- Purely additive/regression-safe: running the script with no CLI arguments at all -- including
  Spyder's "Run file", where `sys.argv` carries no flags -- reproduces the previous hardcoded
  `Control_Var` defaults exactly (verified in `tests/test_quickstart_cli.py`).
- `solete.postprocess.error_msg` gained three new keys (`config_file_not_found`,
  `invalid_config_json`, `invalid_train_val_test`) for CLI/config-file validation errors, following
  the function's existing print-guidance-then-exit convention instead of ad hoc messages.

### Added — model-serving path (`solete.modeling.load_model_bundle`, `solete.inference`)
- `PrepareMLmodel` (`solete/modeling.py`) now accepts an optional `scaler` kwarg; when given (and
  `saveMLmodel`/`trainVSimport` are True, as before), it persists the exact fitted `Scaler`
  PreProcessDataset returned, plus `PossibleFeatures`/`IntrinsicFeature`/`H`/`PRE`/`resolution`/
  `Scaler` type, into a new sidecar `outputs/Trained_<MLtype>_manifest.joblib`. Previously only the
  raw model was saved, and nothing recorded what that model expected or with what scaler it was
  trained -- reloading it for inference meant re-running `PreProcessDataset` on new data, which
  silently *refits* a brand-new scaler instead of reusing the one the model actually learned with.
  Purely additive: omitting `scaler` reproduces the old save behavior exactly (model saved, no
  manifest), and existing `Trained_<MLtype>.joblib`/`.h5` filenames/formats are unchanged.
- New `solete.modeling.load_model_bundle(control)` loads a model together with its manifest and
  validates `PossibleFeatures`/`IntrinsicFeature`/`H`/`PRE`/`resolution`/`Scaler` against `control`
  before returning, raising a clear `ValueError` (not a bare shape-mismatch exception, or worse, a
  silent wrong-semantics "success") on any mismatch.
- New `solete/inference.py`: `build_inference_features()` windows/scales new, already-`ExpandSOLETE`'d
  rows the same way `PreProcessDataset`'s X-side does (same `series_to_forecast` helper), applying
  the bundle's *loaded* scaler (`.transform()` only, never refit); `predict_with_bundle()` runs the
  model and inverse-transforms back to physical units. This is the actual "trained model + new rows
  -> forecast" path that didn't exist before: even `MLForecasting.py` with `trainVSimport=False`
  still re-ran the full training-split pipeline just to assemble inputs for `TestMLmodel`.
- New `scripts/quickstart/predict_with_saved_model.py`: a standalone worked example of the above
  (load a model saved by `MLForecasting.py`, score held-out rows, print the forecasts).
- New `tests/test_modeling_persistence.py`: save -> load -> predict round trip (RF), asserting the
  reloaded model/scaler reproduce the in-memory ones exactly and `predict_with_bundle()`'s output
  matches a hand-built reference window; plus the incompatible-`control`/missing-manifest error
  paths. Closes a zero-coverage gap (`PrepareMLmodel`'s save/load branches and `TestMLmodel` had no
  functional test before this).
- Verified against the real v4 60min data file (RF and a 3-epoch CNN): `predict_with_bundle()`'s
  forecast for the dataset's last row matched `TestMLmodel`'s own forecast for that row exactly, for
  both model families.

### Changed — `solete.dataset.SOLETE` convenience class
- Generalized `SOLETE` beyond its original single-resolution (60min), single-horizon (1h) scope:
  `resolution` now accepts any of `1sec`/`1min`/`5min`/`60min` (`solete.paths.RESOLUTIONS`), and
  `horizon` accepts an int step count or any `pandas.Timedelta`-parseable string (e.g. `"1h"`,
  `"30min"`), scaled to the chosen resolution's native step; a horizon that isn't an exact multiple
  of that step raises `ValueError` instead of silently rounding. The recent-history lag features stay
  `(1, 2, 3)` native steps; the "same time yesterday" lag now scales with resolution (`86400 /
  step_seconds`) instead of a fixed `24`. The default (`resolution="60min"`, `horizon="1h"`) is an
  exact behavioral no-op: it reproduces the original fixed `LAGS=(1,2,3,24)` exactly.
- Added a `data_version` parameter (`"v3"` default, matching every existing caller; `"v4"` reads the
  current figshare release files instead) to `SOLETE.__init__` and `solete.benchmark.common.load_full_df`,
  since an installation that only downloaded the v4 dataset release otherwise has no v3 file for this
  class to find at any resolution.
- Verified against the real v4 data files at all four resolutions, including the full 1-second file
  (slow, ~2.5 minutes, but functions correctly); the automated test suite covers 60min/5min/1min only
  (1sec is impractical to run routinely).
- `solete.benchmark.common.split_df` now slices train/val/test as half-open intervals
  (`[block_start, next_block_start)`) instead of each block's own inclusive "end" timestamp from
  `benchmarks/splits/v1.json`. Identical to the old behavior at 60min (the only resolution v1.json's
  row counts were computed for -- every pair of adjacent blocks already abuts with exactly one 60min
  step of gap); without this fix, sub-hourly resolutions would silently drop every row between a
  block's old hourly "end" and the next block's start into neither block.
- `solete/dataset.py`'s single-split (`v1`) and single-feature-set scope are unchanged by design: a
  second split version is a maintainer data decision, not something this class should default its way
  into (see the module docstring).

### Changed — `dataset/pipeline/build_release.py` orchestration
- Added a `--jobs N` flag (default 1, sequential, unchanged behavior). When `N > 1`, the three stages
  that loop sequentially over slices of the 39.48M-row 1-second file -- `clean` (`clean_solete_1sec.
  clean_file`), `resample` (`resample_solete.resample_file`), and the 1-second half of `expand`
  (`release_stages.stage_expand_1s`) -- compute their slices in a `ProcessPoolExecutor` of that size
  instead of one at a time; each worker reads only its own row range from disk (never the whole file)
  and results are written/concatenated back in original slice order (`ProcessPoolExecutor.map`
  preserves input order even though workers may finish out of order), so output is identical to
  `--jobs 1` for any `N`. Resolution-level parallelism (the coarse `expand`/`parquet` loops) was found
  to be low-value (the three coarse files are 0.3-1.7% the row count of the 1-second file) and is out
  of scope. `--jobs` is also exposed directly on `clean_solete_1sec.py --jobs` and
  `resample_solete.py --jobs` for standalone use; `--dry-run` prints a rough extra-RAM estimate for
  `--jobs > 1`. Stage-level checkpointing/resume (`--skip-existing`) was already in place and is
  unchanged.

## [4.0] - 2026-10-10
This entry is unbearably long because it describes a lot of work done over the spam of a few months.

### Added — v4 dataset release pipeline
- Added `dataset/pipeline/build_release.py` to build the ten HDF5 and Parquet release files, checksums, manifest, methodology and build summary in one command. Supports resumable stages, dry runs, overwrite protection and disk-space checks. Failed or invalidated builds remove stale checksums and manifests.
- Added `make_original.py` to generate `SOLETE_Pombo_1sec_original_v4.h5` from the raw v3 1-second file, preserving the nine measured columns exactly.
- Added `release_verify.py` to check timestamps, row counts, columns, dtypes, cleaning, resampling, expansion, Parquet round trips and reproducibility against the original data. The first full build passed all 28 checks.
- Processing is chunked where possible: cleaning at rule-safe boundaries, resampling by whole days, expansion by month, and streamed Parquet export. HDF5 row-range reading is implemented in `solete/h5io.py`.
- Parquet export now uses `timestamp[ns, tz=UTC]`, million-row groups, statistics, richer metadata and optional compression trials.

### Changed — v4 format and processing
- Standardized v4 filenames as `SOLETE_Pombo_<resolution>_v4.<ext>`, using `60min` rather than `1h`; the original 1-second file has its own `_original` suffix. Centralized path resolution in `solete/paths.py`.
- Removed the constant `Azimuth[deg]_qc` and `Elevation[deg]_qc` columns (previously always code 8). Code 8 remains reserved. The 1-second release has 28 columns; coarser files have 36.
- `clean_solete_1sec.py` now requires chronologically sorted input. The `--in-memory` option retains the previous whole-file processing path.
- Made CoolProp a lazy dependency and moved `import_PV_WT_data` to `solete/params.py`, allowing dataset builds without CoolProp or scikit-learn.

### Changed — QC and expansion
- Added expansion, QC and synthetic diagnostic tests. Then, consolidated QC codes in `solete/qc_codes.py`. Code 6 (`QC_MODEL_SUBSTITUTED`) remains platform-owned; code 11 (`QC_UNTREATED_IMPLAUSIBLE`) applies to legacy v3 raw-value checks. V3 hourly flags retain their previous flagged rows, with the documented code changes.
- Added `solete/expansion.py` with chunkable, resolution-independent and idempotent physical calculations. Measured `P_Solar[kW]` is preserved; `P_Solar_clean[kW]` stores the cleaned series.
- Model-derived columns are recomputed per resolution and excluded from resampling. Removed redundant substitution indicators and QC aliases; `P_hybrid[kW]_qc_source` is now an `int8` code.

### Changed — repository structure and compatibility
- Merged the dataset pipeline and forecasting platform into one repository. The `SOLETE/` package centralizes data handling, physics, QC, metrics and forecasting (see `CONTRIBUTING.md`'s "Code layout" section for the current structure).
- Added `SOLETE/paths.py` for data and output resolution, `pyproject.toml`, `.gitignore` rules and installation documentation. Dataset files are expected under `data/hdf5/` and `data/parquet/`, not in Git.
- Added v4 loading support and retained legacy import paths through a compatibility shim. Split the former monolithic `Functions.py` into modules for QC, physics, preprocessing, I/O, modeling and post-processing.

### Added — forecasting and interoperability
- Added hybrid wind/solar forecasting and ramp-rate analyses, with an example notebook. Results are documented as methodological deliverables rather than evidence that joint forecasting improves performance.
- Added probabilistic PV and wind forecasting using LightGBM quantile regression, with pinball loss, approximate CRPS, interval coverage and sharpness metrics. For giggles.
- Added `r/load_solete.R` for loading the original HDF5 files in R. Tested against both source files; values and datetime indices match pandas output exactly.
- Added tests for hybrid QC inheritance and package import isolation. The full suite passes: 62 tests.

### Fixed — legacy compatibility and numerical performance
- Fixed the inverter-capacity clamp, pandas Series indexing in the thermodynamic model, and scikit-learn metric calls removed in newer versions. Updated the post-processing code to use explicit positional indexing.
- Refactored `Rincon_Pombo_ThermodynamicModel` to batch CoolProp calculations, reducing runtime on the 60-minute dataset by a 2.6× factor, with bit-for-bit identical output on both source files.
- Fixed sample paths and Colab download destinations in the example notebooks. All five current notebooks execute without error.
- Regenerated non-TensorFlow benchmark results affected by the revised QC rule. 

### Changed — ML quickstart (`scripts/quickstart/MLForecasting.py`, `solete/preprocessing.py`, `solete/modeling.py`)
- Added `Control_Var['RF_SVM_use_val_as_train']` toggle: `True` (default) trains RF/SVM on train+val, `False` matches the ANN models' train-only split for a strictly apples-to-apples comparison.
- Added `Control_Var['random_seed']` (default `None`): set an int to make ANN weight initialization/shuffling reproducible via `keras.utils.set_random_seed`.
- Added a per-model `EarlyStopping` sub-dict (`enabled`/`patience`/`restore_best_weights`) for LSTM/CNN/CNN_LSTM, enabled by default.
- Fixed the naive persistence baseline: `PreProcessDataset` now builds it directly from the raw (unscaled) series and carries it row-aligned with `Y_TEST` through every split/filter step, instead of extracting it from `Xscaler`-scaled test data and inverse-transforming with `Yscaler` — an implicit, unchecked assumption that the two scalers agreed on that column. Removed the now-unused `generate_persistence()`.
- Removed the inert `Masking(mask_value=999, ...)` layers from `train_LSTM`/`train_CNN_LSTM` (no 999 placeholder is ever produced; NaN rows are dropped outright) and the unused `features` variable from all three `train_*` functions.

### Fixed — release template
- Restored the `[[SIZES-BEGIN]]`/`[[SIZES-END]]` markers in `data/figshare_README.txt` (the checked-in template `build_release.py`'s `write_manifest()` fills in with real file sizes on every build). A past real build's output — baked-in sizes for the full 39.48M-row dataset, no markers — had been committed over the reusable template, so `write_manifest()` silently skipped writing a fresh `figshare_README.txt`/adding it to `manifest.json`/`SHA256SUMS.txt` on every subsequent build (the code only acts "if markers in text"). Caught by `tests/test_release_build.py::test_build_release_end_to_end_resume_refuse_and_tamper`, which was failing for this reason; `pytest` was not installed in `solete-env` so this had gone unnoticed. Fixed by reinstating the markers around the (now-removed) hardcoded table; the real sizes are unchanged in git history and in `manifest.json`/`SHA256SUMS.txt` for anyone who needs that one snapshot.

### Added — development environment
- Added `pytest==9.1.1` to `requirements.txt` (it was already listed, correctly, as an optional dependency in `dataset/requirements.txt`, but missing from the root one) and installed `pytest` and `pyarrow` into `solete-env`. Full suite now passes: 92 passed, 35 skipped (skips need the real data files, not present here).

### Changed — documentation navigation
- Added a "What SOLETE actually is" summary and a full documentation map to `README.md`, listing every doc file in the repo by audience (start here / dataset / forecasting / historical).
- Created `dataset/README.md` (previously referenced from three places — root `README.md`, `dataset/pipeline/release_stages.py`'s docstring — but never created).
- Moved the stale, pre-v4 `docs/RESOLUTIONS.md` (its "no resampling code anywhere in this repo" claim predates `resample_solete.py`) to `docs/legacy/RESOLUTIONS_v3_investigation.md` with a header marking it historical/superseded by `dataset/docs/METHODOLOGY.md`; updated its six references across `BENCHMARKS.md`, `benchmarks/splits/README.md`, `KNOWN_ISSUES.md`, `docs/legacy/DATA_DICTIONARY_platform_v3.md` and `examples/05_hybrid_forecasting.ipynb`.
- Fixed dead links to `docs/RESTRUCTURE_NOTES.md` (deleted in a past reorg; its still-relevant code-6/QC-vocabulary decision is now in `dataset/docs/QC_SCHEMA.md` §1) and `releases/v3.0_notes.md` (never existed; the corrigendum text is in `README.md`'s own "Corrigendum" section) — six occurrences across `CHANGELOG.md`, `benchmarks/splits/README.md`, `docs/legacy/{DATA_DICTIONARY,QC_SCHEMA}_platform_v3.md`, `solete/paths.py` and `tests/test_dataset_api.py`.

### Known issues
- `P_Gaia[kW]` is zero for 99.56% of the full record, limiting the interpretation of wind and hybrid forecasting results. The cause remains unresolved, the most probable reasons are out-of-service Turbine, or faulty data adquisition.
- Legacy callers must include generated QC columns in `Control_Var['PossibleFeatures']` to preserve them through save/import round trips.
- The full LSTM/RF/SVR training pipeline has been rerun but not used again to properly deploy forecasters.

## [3.0] - 2023-07-20
### Fixed
- Train/validation/test data-splitting bug (**corrigendum**: affected all results computed with v2.3 or earlier).
- RMSE calculation bug in results postprocessing (**corrigendum**: affected all results computed with v2.3 or earlier).
- Error computation and postprocessing improved across all ML models as part of the same fix.

### Changed
- Data preprocessing now adapts to different combinations of previous-samples and forecast horizons, for both ensemble and ANN methods.
- Memory handling improved; code simplified.
- Documentation enhanced.

> **Correction from v2.3:** the train/val/test split and the RMSE calculation both contained bugs, present through v2.3 and fixed here in v3.0. Results computed with v2.3 or earlier are not directly comparable to v3.0+. See the "Corrigendum" section of [README.md](README.md).

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
