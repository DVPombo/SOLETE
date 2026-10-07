# SOLETE dataset — cleaning and quality-control pipeline

Code and documentation behind **version 4** of the SOLETE dataset: 15 months
(2018-06-01 → 2019-08-31) of co-located meteorology, wind-turbine power and
PV power from DTU SYSLAB, Denmark, at 1 s, 1 min, 5 min and 1 h resolution.

This folder is the `dataset/` half of the SOLETE repository (the forecasting platform is in `../solete/` and `../benchmarks/`).
The data files are **not in the repository**. They are published on figshare
(DTU Data): <https://doi.org/10.11583/DTU.17040767>. Put them in `../data/` as described in [`../data/README.md`](../data/README.md). Original dataset paper:
<https://doi.org/10.1016/j.dib.2022.108046>.

## Which file should I use?

| Format | Use it for |
|---|---|
| **Parquet** (`SOLETE_Pombo_*_v4.parquet`) | Analysis and modelling. The cleaned data with quality flags, in a format every language can read. |
| **HDF5** (`.h5`) | Transparency and reproduction. The raw 1-second file, the cleaned file and the resampled files exactly as this pipeline produces them. |

Both formats contain the same numbers. Every value that was changed or is
doubtful carries a quality flag — nothing is silently fixed.

## Quick start

```python
import pandas as pd

df = pd.read_parquet("data/parquet/SOLETE_Pombo_60min_v4.parquet").set_index("timestamp")
ok = df[df["GHI[kW1m2]_qc_frac_flagged"] < 0.1]                          # hours that are < 10 % flagged
```

Column meanings, units and flag codes: [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).
Things worth knowing before you model anything: pressure is valid on a single day,
turbine power is confirmed real on two days, and azimuth/elevation are computed,
not measured.

## Reproduce the cleaning

```bash
# from the repository root, using the prepared environment
examples/.venv/solete-full-template/Scripts/python.exe dataset/pipeline/build_release.py \
  --raw data/hdf5/SOLETE_Pombo_1sec.h5 --slice-days 1
```

Input is the raw 1-second file from version 3 (`SOLETE_Pombo_1sec.h5`, key `DATA`).
The builder streams day-aligned slices, runs each heavy stage in a subprocess,
writes temporary files before atomic replacement, and prints elapsed time and
peak RSS per stage. Compare its JSON summary with
[`docs/CLEANING_DECISIONS.md`](docs/CLEANING_DECISIONS.md).

## Repository layout

```
dataset/pipeline/       the code that produces the released files
  build_release.py       one-command v4 release build and verification
  clean_solete_1sec.py   raw 1 s file -> sorted, cleaned, flagged 1 s file
  resample_solete.py     cleaned 1 s file -> 1 min / 5 min / 1 h files
  export_parquet.py      any of the .h5 files -> .parquet
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
