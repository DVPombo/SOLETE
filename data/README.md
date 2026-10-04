# data/ — put the SOLETE data files here

The data are **not stored in git**. They are published on figshare / DTU Data:

> **https://doi.org/10.11583/DTU.17040767**  (SOLETE dataset, version 4)

Download the files and unzip them **into this folder, keeping the sub-folders**, so you end up with:

```
data/
├── hdf5/
│   ├── SOLETE_Pombo_1sec.h5      original 1-second file (v3, unchanged) – start of the cleaning
│   ├── SOLETE_clean_1sec.h5      cleaned 1 s data with quality flags
│   ├── SOLETE_clean_1min.h5
│   ├── SOLETE_clean_5min.h5
│   └── SOLETE_clean_1h.h5        (what version 3 called 60min)
├── parquet/
│   ├── SOLETE_clean_1sec.parquet
│   ├── SOLETE_clean_1min.parquet
│   ├── SOLETE_clean_5min.parquet
│   └── SOLETE_clean_1h.parquet
└── derived/                      created by the code (caches). Safe to delete.
```

This is exactly the layout of the figshare upload, so no renaming is needed. Files placed directly in
`data/` (without `hdf5/` / `parquet/`) are found as well.

## How the code finds the files

Everything goes through [`solete/paths.py`](../solete/paths.py) — nothing depends on the folder you run a script from.

```python
from solete.paths import find_data_file
import pandas as pd

df = pd.read_parquet(find_data_file("1h", version="v4", fmt="parquet"))   # cleaned, with flags
df = pd.read_hdf(find_data_file("60min", version="v3"))                    # v3 hourly, used by the platform/benchmarks
```

Check what the code sees:

```bash
python -m solete.paths
```

Command-line tools accept a bare file name and look it up here:

```bash
python dataset/pipeline/clean_solete_1sec.py SOLETE_Pombo_1sec.h5      # reads data/hdf5/, writes data/hdf5/SOLETE_clean_1sec.h5
python dataset/pipeline/resample_solete.py   SOLETE_clean_1sec.h5
python dataset/pipeline/export_parquet.py    SOLETE_clean_1h.h5        # writes data/parquet/SOLETE_clean_1h.parquet
```

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
