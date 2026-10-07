# data/ — put the SOLETE data files here

The data are **not stored in git**. They are published on figshare / DTU Data:

> **https://doi.org/10.11583/DTU.17040767**  (SOLETE dataset, version 4)

Download the files and unzip them **into this folder, keeping the sub-folders**, so you end up with:

```
data/
├── hdf5/
│   ├── SOLETE_Pombo_1sec_original_v4.h5
│   ├── SOLETE_Pombo_1sec_v4.h5
│   ├── SOLETE_Pombo_1min_v4.h5
│   ├── SOLETE_Pombo_5min_v4.h5
│   └── SOLETE_Pombo_60min_v4.h5
├── parquet/
│   └── the same five stems with .parquet
└── derived/                      created by the code (caches). Safe to delete.
```

This is exactly the layout of the figshare upload, so no renaming is needed. Files placed directly in
`data/` (without `hdf5/` / `parquet/`) are found as well.

## How the code finds the files

Everything goes through [`solete/paths.py`](../solete/paths.py) — nothing depends on the folder you run a script from.

```python
from solete.paths import find_data_file
import pandas as pd

df = pd.read_parquet(find_data_file("60min", version="v4", fmt="parquet"))
df = pd.read_hdf(find_data_file("60min", version="v3"))                    # v3 hourly, used by the platform/benchmarks
```

Check what the code sees:

```bash
python -m solete.paths
```

The v4 `_original` file is the raw v3 1-second data sorted chronologically,
restricted to the nine measured columns. Values are unchanged. The published
`Azimuth[deg]` and `Elevation[deg]` columns are omitted because the pipeline
recomputes them from UTC timestamps and site coordinates.

Build all ten release files, checksums and the manifest in one command:

```bash
examples/.venv/solete-full-template/Scripts/python.exe dataset/pipeline/build_release.py --raw data/hdf5/SOLETE_Pombo_1sec.h5 --slice-days 1
```

In Spyder, open `dataset/pipeline/build_release.py`, choose **Run >
Configuration per file > Execute in an external system terminal**, and put
`--raw data/hdf5/SOLETE_Pombo_1sec.h5 --slice-days 1` in the command-line
options field. Add `--overwrite` only when intentionally replacing a prior
build; use `--skip-existing` to resume completed stages.

## Keeping the data somewhere else

Set the environment variable `SOLETE_DATA_DIR` to a folder with the same `hdf5/` + `parquet/` structure
(for example an external drive or a cluster scratch directory):

```bash
export SOLETE_DATA_DIR=/mnt/bigdisk/solete        # Linux / macOS
set SOLETE_DATA_DIR=D:\solete                      # Windows cmd
```

In Spyder: *Tools → Preferences → Python interpreter → Environment variables*, or set it at the top of your script
with `os.environ["SOLETE_DATA_DIR"] = ...` **before** importing `solete`.

## Which version does the platform read?

The forecasting platform and the benchmarks were built on the **v3 hourly file** (`SOLETE_Pombo_60min.h5`) and still
read it by default. The cleaned v4 files are what the dataset pipeline produces and what you should use for new
analysis; see `docs/RESTRUCTURE_NOTES.md` for what remains before the platform consumes them.
