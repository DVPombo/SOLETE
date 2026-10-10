# SOLETE dataset — cleaning and quality-control pipeline

Code and documentation behind **version 4** of the SOLETE dataset: 15 months
(2018-06-01 → 2019-08-31) of co-located meteorology, wind-turbine power and
PV power from DTU SYSLAB, Denmark, at 1 s, 1 min, 5 min and 60 min resolution.

This folder is the `dataset/` half of the SOLETE repository (the forecasting platform is in `../solete/` and `../benchmarks/`).
The data files are **not in the repository**. They are published on figshare
(DTU Data): <https://doi.org/10.11583/DTU.17040767>. Put them in `../data/` as described in [`../data/README.md`](../data/README.md). Original dataset paper:
<https://doi.org/10.1016/j.dib.2022.108046>.

## Which file should I use?

| Format | Use it for |
|---|---|
| **Parquet** (`SOLETE_Pombo_<res>_v4.parquet`) | Analysis and modelling. The cleaned data with quality flags and the model columns, in a format every language can read. |
| **HDF5** (`.h5`) | Transparency and reproduction. The same files as HDF5, plus `SOLETE_Pombo_1sec_original_v4`: the raw 1-second data, sorted, nothing cleaned. |

Both formats contain the same numbers. Every value that was changed or is
doubtful carries a quality flag — nothing is silently fixed.

## Quick start

```python
import pandas as pd

df = pd.read_parquet("data/parquet/SOLETE_Pombo_60min_v4.parquet").set_index("timestamp")   # timestamps are UTC
ok = df[df["GHI[kW1m2]_qc_frac_flagged"] < 0.1]                          # hours that are < 10 % flagged
```

Column meanings, units and flag codes: [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).
Things worth knowing before you model anything: pressure is valid on a single day,
turbine power is confirmed real on two days, and azimuth/elevation are computed,
not measured.

## Building the v4 release

```bash
# from the repository root; every path comes from solete/paths.py (data/ or $SOLETE_DATA_DIR)
pip install -r dataset/requirements.txt
python dataset/pipeline/build_release.py --dry-run                      # the plan and the disk estimate, writes nothing
python dataset/pipeline/build_release.py --raw <folder>/SOLETE_Pombo_1sec.h5 --skip-existing
```

One command produces all ten files (`SOLETE_Pombo_1sec_original_v4`, `SOLETE_Pombo_{1sec,1min,5min,60min}_v4`, each as `.h5` and `.parquet`),
`SHA256SUMS.txt`, `manifest.json`, the generated resampling methodology and `build_summary_v4.json`. Stages (`--stages`): `original`
(raw v3 → sorted `_original`), `clean`, `resample`, `expand`, `parquet`, `verify`, `manifest`. Each heavy stage runs in its own
subprocess; nothing is overwritten without `--overwrite`; `--skip-existing` resumes an interrupted build. The Spyder recipe is in the header
of [`pipeline/build_release.py`](pipeline/build_release.py). The raw v3 file is read from `--raw` and never modified; someone who only has
`SOLETE_Pombo_1sec_original_v4.h5` runs `--stages clean,resample,expand,parquet,verify,manifest` (site constants and conventions:
[`../data/README.md`](../data/README.md)).

**Memory.** Everything is sliced (about a month of 1-second rows per slice, `--slice-days`), so no stage holds the 1-second frame. The
cleaning rules look at runs of consecutive rows (pressure flatline, dropout and glitch runs), so slices are cut only at a row boundary that
no run can cross (`clean_solete_1sec.find_safe_cut`); the sliced result is identical to the whole-file result
(`tests/test_release_build.py` proves it with runs placed across the cut points). Peak memory per stage is printed in the final summary.

**Verification.** The `verify` stage prints a table and fails loudly: grid and row counts, column sets, dtypes, `_original` equal to the raw
file, idempotence of `expand_physical`, measured columns equal to the resample of the cleaned data, a reproducibility rebuild from `_original`
(`--rebuild-check sample|full|none`), the platform import, the cleaning counts to compare with
[`docs/CLEANING_DECISIONS.md`](docs/CLEANING_DECISIONS.md), and the Parquet round trip.

After a build on the real file, replace the synthetic effect table in `docs/METHODOLOGY.md` with
`python scripts/expansion_checks.py effect --input <a cleaned 1 s slice of the real build>`.

Single steps, for experiments (the build runs exactly these functions):

```bash
python dataset/pipeline/make_original.py                       # raw v3 -> SOLETE_Pombo_1sec_original_v4.h5
python dataset/pipeline/clean_solete_1sec.py SOLETE_Pombo_1sec_original_v4.h5 --out-prefix SOLETE_Pombo_1sec_cleaned
python dataset/pipeline/resample_solete.py   SOLETE_Pombo_1sec_cleaned.h5 --out-prefix SOLETE_resampled
python dataset/pipeline/export_parquet.py    SOLETE_Pombo_60min_v4.h5 --trial      # measure the compression options
```

The cleaning and resampling scripts print a JSON summary at the end — compare it with
[`docs/CLEANING_DECISIONS.md`](docs/CLEANING_DECISIONS.md). They produce the pre-expansion intermediates; the released
files are the ones `build_release.py` writes after `expand_physical`.

## Repository layout

```
dataset/pipeline/       the code that produces the released files
  build_release.py       the one command: all stages, subprocesses, verification, manifest
  make_original.py       raw v3 1 s file -> sorted _original (nine measured columns)
  clean_solete_1sec.py   _original -> cleaned, flagged 1 s file (every cleaning rule is in clean_block)
  resample_solete.py     cleaned 1 s file -> 1 min / 5 min / 60 min files
  export_parquet.py      any of the .h5 files -> .parquet (pinned ns/UTC timestamp, metadata, round trip)
  release_stages.py      the stages as functions of explicit paths
  release_verify.py      the verification checks
  release_meta.py        per-column provenance and file metadata
  ../solete/h5io.py      slice-wise HDF5 reading/writing (table and fixed format)
  qc_flags.py            flag codes and helpers
  solar_position.py      azimuth/elevation (pvlib)
  solete_report.py       prints the JSON summaries
docs/           what was done, and why
  DATA_DICTIONARY.md     columns, units, flag codes — start here
  CLEANING_DECISIONS.md  every cleaning rule, its reasoning, open questions
  QC_SCHEMA.md           flag codes, precedence and detection rules
  METHODOLOGY.md         resampling conventions
diagnostics/    read-only investigation scripts behind the findings (see its README)
AGENTS.md       orientation for AI coding agents
```

## Citation and licence

Please cite the Data in Brief article (see `CITATION.cff`) and the dataset
version you used. Code and documentation: MIT, see [`LICENSE`](../LICENSE).
