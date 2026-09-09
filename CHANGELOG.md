# Changelog

All notable changes to the SOLETE platform are documented in this file.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Entries for v1.0 through v3.0 were backfilled from the git history and condensed from [releases/](releases/); see those files for more detail, and see `releases/v3.0_notes.md` for the full corrigendum text.

## [Unreleased]
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
