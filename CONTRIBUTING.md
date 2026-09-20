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
  bundled/reference `.h5` files (`SOLETE_short.h5`, `SOLETE_Pombo_60min.h5`, and the
  `examples/` sample files), not synthetic data, wherever practical. If you're fixing a
  bug or reporting a data issue, show it against a real file.
- **Every behavior-affecting change gets a `CHANGELOG.md` entry** under `[Unreleased]`.
  Follow the existing format (see the current `[Unreleased]` section for the level of
  detail expected — what changed, why, and what you verified it against).
- **`QC_SCHEMA.md` and `KNOWN_ISSUES.md` are living documents.** If you find a new
  data-quality issue while using or extending the dataset, add it there following the
  existing entry format (what it is, where it lives, current status, what a user should
  do in the meantime) rather than silently working around it or fixing it without a
  record.

## Code layout

The actual pipeline logic (I/O, QC flagging, physics models, preprocessing, ML
model training, postprocessing) lives in the `solete_pipeline/` package, split
into one module per concern:

- `solete_pipeline/qc.py` — QC flag constants and `apply_qc_flags`/
  `build_*_qc_rules`
- `solete_pipeline/physics.py` — `PV_Performance_Model`,
  `Rincon_Pombo_ThermodynamicModel`
- `solete_pipeline/preprocessing.py` — `ExpandSOLETE`, `PreProcessDataset`,
  `series_to_forecast`
- `solete_pipeline/io.py` — `import_SOLETE_data`, `import_SOLETE_sample`,
  `import_PV_WT_data`
- `solete_pipeline/modeling.py` — `PrepareMLmodel`, `train_LSTM`/`train_CNN`/
  `train_CNN_LSTM`, `TestMLmodel`, `generate_persistence`
- `solete_pipeline/postprocess.py` — `post_process`, `error_msg`

`Functions.py` at the repo root still exists and still works exactly as
before (`from Functions import X` for any of the names above) — it's now a
thin re-export shim over `solete_pipeline`, kept for backward compatibility
with every existing script and notebook. **New code should import from the
specific `solete_pipeline` submodule it needs, not from `Functions.py`** —
e.g. `from solete_pipeline.qc import apply_qc_flags` rather than
`from Functions import apply_qc_flags`. This matters beyond style: importing
`solete_pipeline.qc` or `solete_pipeline.physics` directly costs nothing
beyond `pandas`/`numpy`, while `Functions.py` still imports everything,
including `keras`/`tensorflow` for the ML training functions, since it has
to keep re-exporting those too.

This split was done in Phase 7, Session 8, specifically to make individual
pieces (especially the QC layer and the physics models) unit-testable in
isolation — see `tests/test_solete_pipeline_units.py` for the isolation
tests this enabled, and the module-splitting motivation generally.

## Running the tests

Tests live in `tests/` and use `pytest`:

```
pip install pytest
pytest tests/ -v
```

Run from the repo root. If you add a new QC rule, data-fixing behavior, or anything
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
