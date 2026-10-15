# dataset/ — the cleaning & quality-control pipeline

This folder turns the raw, 15-month, 1-second-resolution SOLETE recording into the
published, quality-controlled **version 4 release** (ten HDF5/Parquet files under
`data/`). If you're an AI coding agent, read [`AGENTS.md`](AGENTS.md) first — it has the
rules (never modify raw files, verify against real data, etc.). This file is the human
landing page: what's here and where to read about it.

## Layout

| Folder | What's in it |
|---|---|
| [`pipeline/`](pipeline/) | The code that produces the release (cleaning, resampling, Parquet export, the `build_release.py` orchestrator) |
| [`docs/`](docs/) | The authoritative documentation of every decision, schema and methodology (see below) |
| [`diagnostics/`](diagnostics/) | Read-only investigation scripts that justified the cleaning decisions — not needed for normal use |

## Documentation (`docs/`)

| Doc | Read it for |
|---|---|
| [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) | Every column: units, which resolution(s) it appears in, provenance |
| [`docs/QC_SCHEMA.md`](docs/QC_SCHEMA.md) | The `<column>_qc` flag vocabulary, precedence order, how flags behave when resampled |
| [`docs/CLEANING_DECISIONS.md`](docs/CLEANING_DECISIONS.md) | What each cleaning rule changed and why, with open questions at the end |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | How resampling/aggregation works (window convention, circular means, model columns computed per resolution) |

Read `CLEANING_DECISIONS.md` first if you want the *why*; `QC_SCHEMA.md` is the lookup table.

## Building the v4 release

From the repository root:

```
pip install -r dataset/requirements.txt
python dataset/pipeline/build_release.py --raw SOLETE_Pombo_1sec.h5 --skip-existing    # all ten v4 files + checksums; --dry-run first
```

`clean`/`resample`/`expand` each process the 1-second file slice by slice; add `--jobs N` to run those
slice loops in `N` worker processes instead of one (results are identical either way -- writes stay in
original slice order). Each extra worker holds one more slice in RAM at a time, so raise `--jobs` only
with cores and RAM to spare (`--dry-run` prints a rough estimate of the extra memory).

`build_release.py` runs the stages in order (each implemented once, elsewhere — this
module only moves slices between them):

```
original  raw v3 1 s            -> SOLETE_Pombo_1sec_original_v4.h5   (make_original.py)
clean     _original              -> cleaned 1 s, QC-flagged            (clean_solete_1sec.py)
resample  cleaned 1 s            -> 1min / 5min / 60min                (resample_solete.py)
expand    every resolution       -> + model columns (Pac, P_hybrid[kW], ...) (solete/expansion.py)
parquet   every HDF5 file        -> matching .parquet file              (export_parquet.py)
```

See [`docs/CLEANING_DECISIONS.md`](docs/CLEANING_DECISIONS.md) and
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for what each stage actually does, and the
root [`README.md`](../README.md) for where the output files end up and how code finds them.
