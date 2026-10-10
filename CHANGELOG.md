# Changelog

All notable changes to the SOLETE platform are documented in this file. From latest to oldest as you scroll down.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Entries for v1.0 through v3.0 were backfilled from the git history; see those files for more detail, and see `releases/v3.0_notes.md` for the full corrigendum text.

## [4.0] - 2026-10-10
This entry is unbearably wrong because it describes a lot of work done over the spam of a few months.

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
- Merged the dataset pipeline and forecasting platform into one repository. The `SOLETE/` package centralizes data handling, physics, QC, metrics and forecasting; `docs/RESTRUCTURE_NOTES.md` records the old-to-new paths.
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
