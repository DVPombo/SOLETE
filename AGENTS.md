# AGENTS.md — orientation for AI coding agents

This repository is the whole SOLETE project: `dataset/` (cleaning and quality control of the dataset, version 4) and
`solete/` + `benchmarks/` (forecasting platform). **Read `dataset/AGENTS.md` before touching anything under `dataset/`.**

Rules that apply everywhere:

1. **Paths.** Never hard-code a data path, never rely on the current directory. Use `solete/paths.py`
   (`find_data_file`, `resolve_input`, `resolve_output_prefix`, `derived_path`, `output_path`). Data lives in `data/` (git-ignored, figshare layout `hdf5/` + `parquet/`). v4 file names are built only in `solete/paths.py` (`data_filename`, `release_path`, `release_stems`); big files are read in slices with `solete/h5io.py`, never loaded whole.
2. **Never modify a data file in place.** Outputs go to new files. Raw inputs are read-only.
3. **Imports.** Import from the specific module (`from solete.qc import ...`). `solete/__init__.py` re-exports nothing; `solete.qc`, `solete.physics`, `solete.paths` must stay free of keras/tensorflow (a test enforces it).
4. **New scripts** start with the repo-root `sys.path` bootstrap used by the existing ones (see `benchmarks/baseline_gbm.py`), so they run from any folder and from Spyder.
5. **Verify against real data**, not only synthetic data, and say plainly what was and was not run.
6. **One QC vocabulary**, defined in `solete/qc_codes.py` (never define a flag number anywhere else). Code 6 is
   platform-owned (`solete/expansion.py` only); pipeline rules never emit it. Measured columns and pipeline flags are
   resampled from 1 s; model-derived columns are computed per resolution and never averaged up
   (`dataset/docs/METHODOLOGY.md`). The released files never overwrite measured `P_Solar[kW]`.
7. Every behaviour-affecting change gets a `CHANGELOG.md` entry under `[Unreleased]`.
8. Run `pytest` from the repo root. Tests needing `data/hdf5/SOLETE_Pombo_60min.h5` skip when it is absent.
