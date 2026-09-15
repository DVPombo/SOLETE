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

## A licensing inconsistency you should know about

`LICENSE` and `CITATION.cff` say MIT, but the header comments in some scripts (e.g.
`Functions.py`) say CC-BY 4.0. This is a known, pre-existing inconsistency that hasn't
been resolved yet. If your contribution touches licensing in any way, or you're unsure
which license governs a specific file, **ask the maintainer before assuming** — don't
try to reconcile it yourself.

## Questions

If GitHub Discussions is enabled on this repo, ask there. Otherwise, open an issue with
the `question` template.
