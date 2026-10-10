# Contributing to SOLETE

SOLETE is a research dataset-plus-code project with one active maintainer (Daniel
Vázquez Pombo), not a large open-source project — this guide is intentionally short.

## Proposing a change

- **Anything non-trivial** (new features, behavior changes, new data-quality findings):
  open an issue first and describe what you want to do before writing code. This avoids
  duplicated work and lets the maintainer weigh in on approach.
- **Small fixes or docs** (typos, broken links, small clarifications): a pull request
  directly is fine, no issue needed first.

## Conventions this repo already follows

Please follow these rather than reinventing your own approach — they're patterns
established over the last several phases of work on this repo:

- **Real-data-first verification.** Findings and fixes are checked against the actual
  bundled/reference `.h5` files (`examples/SOLETE_short.h5`, the v3 hourly file in `data/hdf5/`, and the
  other `examples/` sample files), not synthetic data, wherever practical. If you're fixing a
  bug or reporting a data issue, show it against a real file.
- **Every behavior-affecting change gets a `CHANGELOG.md` entry** under `[Unreleased]`.
  Follow the existing format (see the current `[Unreleased]` section for the level of
  detail expected — what changed, why, and what you verified it against).
- **`dataset/docs/QC_SCHEMA.md` and `KNOWN_ISSUES.md` are living documents.** If you find a new
  data-quality issue while using or extending the dataset, add it there following the
  existing entry format (what it is, where it lives, current status, what a user should
  do in the meantime) rather than silently working around it or fixing it without a
  record.

## Code layout

The repository has two halves that share one data folder and one flag-code vocabulary:

- **`dataset/`** — the cleaning and quality-control pipeline that produced version 4 of the dataset
  (`pipeline/`), the documentation of every decision (`docs/`) and the read-only investigation
  scripts (`diagnostics/`). Read `dataset/AGENTS.md` before changing anything there: the rules
  about never modifying a data file and about verifying against real data apply.
- **`solete/`** — the importable package behind the forecasting platform, one module per concern:
  - `paths.py` — **where every file lives** (data folder, examples, outputs). Never build a data path by hand;
    use `find_data_file()`, `resolve_input()`, `output_path()`, ... so nothing depends on the working directory.
  - `qc_codes.py` — the single QC vocabulary; `qc.py` — reading flags, the code-6 substitution flag, legacy v3 checks; `expansion.py` — `expand_physical`
  - `physics.py` — `PV_Performance_Model`, `Rincon_Pombo_ThermodynamicModel`
  - `preprocessing.py` — `ExpandSOLETE`, `PreProcessDataset`, `series_to_forecast`
  - `io.py` — `import_SOLETE_data`, `import_SOLETE_sample`, `import_PV_WT_data`
  - `modeling.py` — `PrepareMLmodel`, `train_LSTM`/`train_CNN`/`train_CNN_LSTM`, `TestMLmodel`, `generate_persistence`
  - `postprocess.py` — `post_process`, `error_msg`
  - `metrics.py`, `dataset.py`, `benchmark/common.py` — metrics, the `SOLETE` convenience class, shared benchmark harness
- **`benchmarks/`**, **`scripts/`**, **`examples/`** — runnable things built on the package.

Import from the specific submodule you need: `from solete.qc import apply_qc_flags`. `solete/__init__.py`
re-exports nothing on purpose, so importing `solete.qc`, `solete.physics` or `solete.paths` costs nothing beyond
`pandas`/`numpy` — only `solete.modeling` pulls in `keras`/`tensorflow`. The dataset scripts rely on this
(they import only light submodules: `solete.paths`, `solete.h5io`, `solete.params`, `solete.qc_codes`, `solete.expansion`, `solete.physics`; CoolProp is imported lazily inside the one thermodynamic model that needs it), and `tests/test_solete_pipeline_units.py` guards it.

Every runnable script starts with a two-line bootstrap that puts the repository root on `sys.path`, so
scripts work from any folder and in Spyder without installing the package. Keep it when you add scripts.

## Running the tests

Tests live in `tests/` and use `pytest`:

```
pip install pytest
pytest tests/ -v
```

Run from the repo root. Tests that need the v3 hourly file (`data/hdf5/SOLETE_Pombo_60min.h5`, not in git) are
**skipped**, not failed, when it is absent. If you add a new QC rule, data-fixing behavior, or anything
else with correctness implications, add a corresponding test in `tests/`, built from a
real row in one of the reference `.h5` files where possible (see the comment at the top
of `tests/test_qc_flags.py` for the pattern, including how the couple of unavoidable
synthetic cases are marked).

## Licensing

This repository is MIT licensed — see `LICENSE` at the repo root, and `CITATION.cff` if
you're citing the software itself. Every script header and `requirements.txt` now says
the same thing consistently. (Previously, several script headers and `requirements.txt`
said CC-BY 4.0 while `LICENSE`/`CITATION.cff` said MIT — a known inconsistency flagged
across Phase 7's Sessions 3 and 4; resolved in favor of MIT per the maintainer's
decision.) If you use this work, the maintainer would appreciate credit (see
`CITATION.cff`) — that's a request, not a license term beyond what MIT itself requires
(keeping the copyright/license notice in copies).

## Questions

If GitHub Discussions is enabled on this repo, ask there. Otherwise, open an issue with
the `question` template.
