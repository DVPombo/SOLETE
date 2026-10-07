SOLETE DATASET VERSION 4
========================

Timestamps are UTC. HDF5 files use a timezone-naive DatetimeIndex whose
values are UTC. Parquet files put timestamp first as timestamp[ns, UTC].
Resampled rows are labelled by interval start: T represents [T, T+period).

FILES
-----
Each HDF5 file has a Parquet counterpart with the same stem and row count.
Exact byte sizes and SHA-256 digests are recorded by the release build in
manifest.json and SHA256SUMS.txt; sizes are build-output facts and are not
predeclared in this source document.

SOLETE_Pombo_1sec_original_v4   39,484,801 rows
  The uncleaned v3 values, chronologically sorted. Contains only the nine
  measured columns needed by the pipeline and platform. Azimuth and elevation
  are omitted because they are recomputed from timestamp and site coordinates.

SOLETE_Pombo_1sec_v4            39,484,801 rows
  Cleaned measured values, pipeline QC flags, recomputed solar position, and
  physical/model columns computed from the cleaned one-second inputs.

SOLETE_Pombo_1min_v4               658,081 rows
SOLETE_Pombo_5min_v4               131,617 rows
SOLETE_Pombo_60min_v4               10,969 rows
  Measured values and pipeline-owned QC fields resampled from cleaned
  one-second data. WIND_DIR uses a circular mean; QC becomes _qc_worst and
  _qc_frac_flagged. Model columns are computed at the named resolution.

The inclusive data span is 2018-06-01 00:00:00 through
2019-09-01 00:00:00 UTC. The final measurement is retained and starts one
additional, single-sample terminal bucket at each coarser resolution.

MODEL COLUMN NOTE
-----------------
The released 1-minute, 5-minute and hourly model columns are not averages of
the one-second model output. The PV model, clipping, and substitution threshold
are nonlinear, so averaging model output answers a different question from
running the model on that interval's cleaned mean inputs. Version 4 therefore
computes Pac, Pdc, module/cell temperatures, clean PV, hybrid power, and code 6
independently at every resolution. Measured P_Solar[kW] is always retained;
P_Solar_clean[kW] is the added substituted series.

REBUILD
-------
From the repository root with the prepared environment:

examples/.venv/solete-full-template/Scripts/python.exe dataset/pipeline/build_release.py --raw data/hdf5/SOLETE_Pombo_1sec.h5 --slice-days 1

The raw v3 file is read-only. The builder writes temporary files, atomically
renames successful outputs, verifies reproducibility and Parquet round trips,
and writes manifest.json and SHA256SUMS.txt.

Full column definitions, provenance and QC codes are in
dataset/docs/DATA_DICTIONARY.md and dataset/docs/QC_SCHEMA.md.