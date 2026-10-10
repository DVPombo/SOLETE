Get the associated code in https://github.com/DVPombo/SOLETE

# data/ — put the SOLETE data files there

Download the files and unzip them **into this folder, keeping the sub-folders**, so you end up with:

```
data/
├── hdf5/
│   ├── SOLETE_Pombo_1sec.h5                 the RAW v3 1-second file (unchanged; only the input of the build)
│   ├── SOLETE_Pombo_1sec_original_v4.h5     the same data, chronologically sorted, nine measured columns, nothing cleaned
│   ├── SOLETE_Pombo_1sec_v4.h5              cleaned + flagged + expanded 1 s data
│   ├── SOLETE_Pombo_1min_v4.h5
│   ├── SOLETE_Pombo_5min_v4.h5
│   └── SOLETE_Pombo_60min_v4.h5             (v3 also called this resolution 60min; "1h" is only an input alias in the code)
├── parquet/
│   └── the same five *_v4 stems with the extension .parquet
└── derived/                                 created by the code (caches, build scratch). Safe to delete.
```

This is exactly the layout of the figshare upload, so no renaming is needed. Files placed directly in
`data/` (without `hdf5/` / `parquet/`) are found as well. The v3 files (`SOLETE_Pombo_<res>.h5`) keep their names and
are not part of the v4 upload; the platform and the benchmarks still read the v3 hourly file by default.

## What each v4 file is

| file | what it holds |
|---|---|
| `SOLETE_Pombo_1sec_original_v4` | The raw 1-second data of v3: **nothing cleaned, no derived column**, the nine measured columns only (`TEMPERATURE[degC]`, `HUMIDITY[%]`, `WIND_SPEED[m1s]`, `WIND_DIR[deg]`, `GHI[kW1m2]`, `POA Irr[kW1m2]`, `P_Gaia[kW]`, `P_Solar[kW]`, `Pressure[mbar]`). It differs from the v3 file in two ways only: the rows are sorted by time (v3 stores the 457 daily blocks in shuffled order) and `Azimuth[deg]` / `Elevation[deg]` are dropped (they are computed from timestamp and site, not measured, and the originals were faulty). Values are bit-identical to v3. It is released so that the cleaning is transparent and reproducible. |
| `SOLETE_Pombo_1sec_v4` | `_original` → pipeline (cleaning, `<column>_qc` flags, **recomputed** Azimuth/Elevation) → `expand_physical` (9 model columns). 28 columns. |
| `SOLETE_Pombo_{1min,5min,60min}_v4` | Measured columns and pipeline flags **resampled from the cleaned 1 s data** (flags become `_qc_worst` + `_qc_frac_flagged`), then `expand_physical` run **on each file from its own inputs**. Hourly model columns are therefore not an average of 1-second model output (`dataset/docs/METHODOLOGY.md`, "Model columns are per resolution"). |

Timestamps are **UTC** everywhere (tz-naive index in HDF5, tz-aware UTC `timestamp` column, unit nanoseconds, in Parquet). Resampled rows are
labelled by the start of the interval `[T, T+period)`. There is no local-time column. Column lists and the provenance of each column:
`dataset/docs/DATA_DICTIONARY.md`; the figshare text is `data/figshare_README.txt`.

## Rebuilding everything from `_original`

Someone who downloads only `SOLETE_Pombo_1sec_original_v4.h5` can regenerate every other file with the pipeline in this repository
(`dataset/pipeline/`). Azimuth and Elevation are recomputed with **pvlib** (NREL SPA) from the timestamp and these site constants
(`SITE_LATITUDE`, `SITE_LONGITUDE`, `SITE_ALTITUDE_M` in `dataset/pipeline/solar_position.py`):

| | |
|---|---|
| latitude / longitude / altitude | 55.6867 N / 12.0985 E / 10 m (DTU Risø campus, SYSLAB) |
| time | UTC |
| `Azimuth[deg]` | 0 = south, east negative, west positive, range [-180, 180) |
| `Elevation[deg]` | apparent elevation, **not** clipped at night (negative below the horizon) |
| pvlib version | written into the Parquet metadata of every file and into `manifest.json` |

The same values are embedded in the Parquet metadata of every file (key `solete`).

```bash
# the whole release, one command (see the header of dataset/pipeline/build_release.py for the Spyder recipe):
python dataset/pipeline/build_release.py --raw <folder>/SOLETE_Pombo_1sec.h5 --skip-existing
# someone who only has _original skips stage 0:
python dataset/pipeline/build_release.py --stages clean,resample,expand,parquet,verify,manifest
```

## How the code finds the files

Everything goes through [`solete/paths.py`](../solete/paths.py) — nothing depends on the folder you run a script from.
The v4 file-name scheme is implemented there and nowhere else (`data_filename`, `release_path`, `release_stems`).

```python
from solete.paths import find_data_file
import pandas as pd

df = pd.read_parquet(find_data_file("60min", version="v4", fmt="parquet"))   # cleaned, with flags ("1h" also works as input)
df = pd.read_hdf(find_data_file("60min", version="v3"))                      # v3 hourly, used by the platform/benchmarks
```

Check what the code sees:

```bash
python -m solete.paths
```

Command-line tools accept a bare file name and look it up here:

```bash
python dataset/pipeline/make_original.py                                          # raw v3 -> SOLETE_Pombo_1sec_original_v4.h5
python dataset/pipeline/clean_solete_1sec.py SOLETE_Pombo_1sec_original_v4.h5     # sliced; writes data/hdf5/SOLETE_Pombo_1sec_cleaned.h5 (an intermediate)
python dataset/pipeline/resample_solete.py   SOLETE_Pombo_1sec_cleaned.h5         # intermediates, before expansion
python dataset/pipeline/export_parquet.py    SOLETE_Pombo_60min_v4.h5             # writes data/parquet/SOLETE_Pombo_60min_v4.parquet
```

(The individual scripts are what `build_release.py` runs for you; use it for the release, the scripts for experiments.)