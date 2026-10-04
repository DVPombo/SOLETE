# AGENTS.md — orientation for AI coding agents

This repository is the whole SOLETE project: `dataset/` (cleaning and quality control of the dataset, version 4) and
`solete/` + `benchmarks/` (forecasting platform). **Read `dataset/AGENTS.md` before touching anything under `dataset/`.**

Rules that apply everywhere:

1. **Paths.** Never hard-code a data path, never rely on the current directory. Use `solete/paths.py`
   (`find_data_file`, `resolve_input`, `resolve_output_prefix`, `derived_path`, `output_path`). Data lives in `data/` (git-ignored, figshare layout `hdf5/` + `parquet/`).
2. **Never modify a data file in place.** Outputs go to new files. Raw inputs are read-only.
3. **Imports.** Import from the specific module (`from solete.qc import ...`). `solete/__init__.py` re-exports nothing; `solete.qc`, `solete.physics`, `solete.paths` must stay free of keras/tensorflow (a test enforces it).
4. **New scripts** start with the repo-root `sys.path` bootstrap used by the existing ones (see `benchmarks/baseline_gbm.py`), so they run from any folder and from Spyder.
5. **Verify against real data**, not only synthetic data, and say plainly what was and was not run.
6. **The two QC code sets differ** (`docs/RESTRUCTURE_NOTES.md` §2). Do not mix them; do not enable `data_version='v4'` in the platform before that is reconciled.
7. Every behaviour-affecting change gets a `CHANGELOG.md` entry under `[Unreleased]`.
8. Run `pytest` from the repo root. Tests needing `data/hdf5/SOLETE_Pombo_60min.h5` skip when it is absent.
