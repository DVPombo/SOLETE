SOLETE dataset, version 4
=========================

15 months (2018-06-01 00:00:00 to 2019-08-31 23:59:59, UTC) of co-located meteorology, wind-turbine power and PV power
from DTU SYSLAB (Risoe campus, Denmark; latitude 55.6867 N, longitude 12.0985 E, altitude 10 m), at 1 s, 1 min, 5 min and
60 min resolution. Code, documentation and the tests that build these files: the SOLETE repository (see the repository URL in the
record description). Dataset record: https://doi.org/10.11583/DTU.17040767
Original dataset paper: https://doi.org/10.1016/j.dib.2022.108046

THE FILES (each exists as HDF5 in hdf5/ and as Parquet in parquet/; same numbers, same column names, same row order)
-----------------------------------------------------------------------------------------------------------------
  SOLETE_Pombo_1sec_original_v4   39,484,800 rows,  9 columns  the RAW 1 s data, sorted, nothing cleaned, nothing derived
  SOLETE_Pombo_1sec_v4            39,484,800 rows, 28 columns  cleaned + quality flags + recomputed sun angles + model columns
  SOLETE_Pombo_1min_v4               658,080 rows, 36 columns  resampled from the cleaned 1 s data, then model columns
  SOLETE_Pombo_5min_v4               131,616 rows, 36 columns  same, 5 minutes
  SOLETE_Pombo_60min_v4               10,968 rows, 36 columns  same, one hour
  (Parquet files have one more column, `timestamp`, as the first column.)

  File sizes (written by the build from the files it produced; see also manifest.json and SHA256SUMS.txt):
[[SIZES-BEGIN]]
  hdf5/SOLETE_Pombo_1sec_original_v4.h5                      467.6 MB
  hdf5/SOLETE_Pombo_1sec_v4.h5                             1,584.1 MB
  hdf5/SOLETE_Pombo_1min_v4.h5                                64.8 MB
  hdf5/SOLETE_Pombo_5min_v4.h5                                14.2 MB
  hdf5/SOLETE_Pombo_60min_v4.h5                                1.3 MB
  parquet/SOLETE_Pombo_1sec_original_v4.parquet              332.7 MB
  parquet/SOLETE_Pombo_1sec_v4.parquet                     1,623.3 MB
  parquet/SOLETE_Pombo_1min_v4.parquet                        47.8 MB
  parquet/SOLETE_Pombo_5min_v4.parquet                        12.4 MB
  parquet/SOLETE_Pombo_60min_v4.parquet                        1.2 MB
[[SIZES-END]]

Which file should I use?  For analysis: the Parquet files (SOLETE_Pombo_60min_v4.parquet is the usual starting point).
For transparency and reproduction: the HDF5 files, in particular SOLETE_Pombo_1sec_original_v4.h5.

TIME
----
All timestamps are UTC. HDF5: tz-naive index whose values are UTC. Parquet: column `timestamp`, timestamp[ns, tz=UTC].
There is no local-time column. Resampled rows are labelled by the START of their interval: a row stamped T covers [T, T+period).

THE `_original` FILE
--------------------
The raw 1-second data of version 3, in chronological order (the version-3 file stores 457 daily blocks in shuffled order). Nine
measured columns, values unchanged: TEMPERATURE[degC], HUMIDITY[%] (a 0-1 fraction despite the name), WIND_SPEED[m1s],
WIND_DIR[deg], GHI[kW1m2], POA Irr[kW1m2], P_Gaia[kW], P_Solar[kW], Pressure[mbar]. The version-3 columns Azimuth[deg] and
Elevation[deg] are dropped: they are computed from the timestamp and the site, not measured, and the old values were faulty.
It contains known problems (placeholder pressure values, wind direction above 360, sensor dropouts, glitches): that is why it is released
untouched, so every cleaning step can be inspected and reproduced.

COLUMN PROVENANCE
-----------------
  measured                    the nine columns above. In `_original`: as recorded. In the cleaned 1 s file: changed only where the column's
                              `_qc` flag is non-zero (P_Solar[kW] and P_Gaia[kW] are never changed). In the 1 min / 5 min / 60 min files: resampled
                              from the cleaned 1 s data (mean; circular mean for WIND_DIR[deg]).
  Azimuth[deg], Elevation[deg]  computed, not measured. Recomputed by the pipeline with pvlib (NREL SPA) from the timestamp and the site
                              (latitude 55.6867, longitude 12.0985, altitude 10 m, UTC). Azimuth: 0 = south, east negative, west positive,
                              range [-180, 180). Elevation: apparent, negative at night (not clipped). In the coarser files: mean of the
                              1 s values (circular mean for the azimuth). There is no flag column for them (it would be constant).
  <column>_qc (1 s file)      int8 quality flag, one code per value: 0 ok, 1 wrapped modulo 360, 2 placeholder set to NaN, 3 dropout short run
                              interpolated, 4 dropout long run kept, 5 glitch short run interpolated, 7 value plausible but cause unknown,
                              9 glitch long run set to NaN, 10 confirmed-good P_Gaia day; 6 is described below; 8 and 11 are not used in v4.
                              Columns flagged: WIND_DIR, Pressure, WIND_SPEED, HUMIDITY, TEMPERATURE, GHI, POA Irr, P_Gaia.
  <column>_qc_worst,          (coarser files) the most severe code among the seconds of the bucket, and the fraction of seconds that were
  <column>_qc_frac_flagged    not code 0. NaN = the bucket has no seconds at all.
  Pac, Pdc, TempModule,       model columns, COMPUTED AT EACH RESOLUTION from that file's own inputs (see the next section):
  TempCell, P_Solar_clean[kW],  King's PV performance model of the 10 kW PV string; P_Solar[kW]_qc (code 6 = the measured PV power is at least
  P_hybrid[kW], P_Solar[kW]_qc, 1.5 times below the model, and the model produces power); P_Solar_clean[kW] (measurement, or the model where
  P_hybrid[kW]_qc,            flagged 6); P_hybrid[kW] = P_Solar_clean[kW] + P_Gaia[kW]; the hybrid flag and where it came from (0 none,
  P_hybrid[kW]_qc_source      1 P_Solar, 2 P_Gaia). Measured P_Solar[kW] is never overwritten.

THE HOURLY MODEL COLUMNS ARE NOT AN AVERAGE OF 1-SECOND MODEL OUTPUT
--------------------------------------------------------------------
In the 1 min, 5 min and 60 min files the model columns are NOT the mean of the 1-second model columns. They are computed again, from that
file's own (already averaged) inputs. This is deliberate: the PV model is not linear (cell temperature, clipping at zero and at the inverter
maximum), and "model is 1.5 times above the measurement" is a threshold that picks different moments at one second than on an hourly mean. So
the hourly Pac is not the mean of the 1-second Pac, and the hourly substitution flag is not the share of flagged seconds. The difference is
small in aggregate but real; its size is tabulated in dataset/docs/METHODOLOGY.md (that table currently describes a synthetic data generator
until it is regenerated from the real files). The 1-second model columns are instantaneous model estimates, not physically validated.

REBUILDING EVERYTHING FROM `_original`
--------------------------------------
Everything else in this record can be regenerated from SOLETE_Pombo_1sec_original_v4.h5 with the repository's pipeline:
    pip install -r dataset/requirements.txt
    python dataset/pipeline/build_release.py --stages clean,resample,expand,parquet,verify,manifest
(the original is read from data/hdf5/; the build runs in slices, a machine with 4 GB of RAM is enough). Azimuth and Elevation are recomputed from the
timestamp with pvlib and the site constants above; the pvlib version used for this release is in the Parquet metadata (key `solete`) and in
manifest.json. The build ends with a verification table that compares the rebuilt files with each other.

WHAT IS KNOWN TO BE UNRELIABLE
------------------------------
  Pressure[mbar]  valid on 2019-01-16 only; every other second was a placeholder and is NaN in the cleaned files.
  P_Gaia[kW]      confirmed real telemetry on 2018-08-31 and 2019-05-25 only; elsewhere the zeros may mean "turbine off" or "not logged".
  HUMIDITY[%]     2018-11-17 is NaN (stuck sensor).
See dataset/docs/DATA_DICTIONARY.md, CLEANING_DECISIONS.md and QC_SCHEMA.md in the repository.

CHECKSUMS
---------
SHA256SUMS.txt lists the SHA-256 of every file; manifest.json lists rows, columns and sizes.
