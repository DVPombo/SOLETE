# SOLETE data dictionary (v4, cleaned files)

Site: SYSLAB, DTU Risø campus, Denmark — latitude 55.6867° N, longitude
12.0985° E, altitude 10 m. Co-located 11 kW Gaia wind turbine, 10 kW PV
system and a weather station.

## Files and time axis

| File family | Rows (expected) | One row is |
|---|---|---|
| `*_1sec` | 39,484,801 | one second; **2018-06-01 00:00:00 → 2019-09-01 00:00:00**, inclusive |
| `*_1min` | 658,081 | one minute |
| `*_5min` | 131,617 | five minutes |
| `SOLETE_Pombo_60min_v4` | 10,969 | one hour |

- **Timestamps are UTC**, not local time (evidence: `CLEANING_DECISIONS.md` §2).
  In the `.h5` files the index is stored tz-naive; in the `.parquet` files it is
  the column `timestamp`, typed `timestamp[ns, UTC]`.
- Resampled rows are labelled by the **start** of their interval, `[T, T+period)`.
- The final `2019-09-01 00:00:00` measurement is retained. It starts one
  additional, single-sample terminal bucket in each coarser file.
- Missing values are `NaN`. The 1-second grid itself has no gaps or duplicates.

`SOLETE_Pombo_1sec_original_v4` contains only the nine measured columns in
the table below. It is the v3 raw 1-second input sorted chronologically, with
values unchanged. Sorting and removal of the published `Azimuth[deg]` and
`Elevation[deg]` columns are its only differences from v3; those two columns
are recomputed by the pipeline. The other four v4 files are expanded release
files. Every HDF5 stem has a matching Parquet file.

## Data columns

For measured columns, "resampled from 1 s" applies to the 1-minute, 5-minute
and 1-hour files; the 1-second file contains the cleaned measurement itself.

| Column | Unit | Meaning | Provenance | Things to know |
|---|---|---|---|---|
| `TEMPERATURE[degC]` | °C | ambient air temperature | measured; resampled from cleaned 1 s | one −40.1 °C second removed as a glitch |
| `HUMIDITY[%]` | **fraction 0–1** | relative humidity | measured; resampled from cleaned 1 s | despite `%` in the name the values are fractions; 2018-11-17 is NaN |
| `WIND_SPEED[m1s]` | m/s | wind speed | measured; resampled from cleaned 1 s | joint zero-dropouts with humidity interpolated if ≤ 5 s |
| `WIND_DIR[deg]` | degrees | wind direction | measured; circularly resampled from cleaned 1 s | wrapped to [0, 360) |
| `GHI[kW1m2]` | kW/m² | global horizontal irradiance | measured; resampled from cleaned 1 s | bound-checked to [0, 1.5] |
| `POA Irr[kW1m2]` | kW/m² | plane-of-array irradiance | measured; resampled from cleaned 1 s | bound-checked to [0, 1.6] |
| `Pressure[mbar]` | mbar (= hPa) | atmospheric pressure | measured; resampled from cleaned 1 s | **valid on 2019-01-16 only**; every other second was a placeholder and is NaN |
| `P_Gaia[kW]` | kW | active power, 11 kW Gaia wind turbine | measured; resampled from cleaned 1 s | only two days are confirmed real telemetry; other zeros have unresolved provenance |
| `P_Solar[kW]` | kW | measured active power, 10 kW PV inverter | measured; resampled from cleaned 1 s | never overwritten in release files |
| `Azimuth[deg]` | degrees | solar azimuth | recomputed by pipeline at 1 s; resampled from 1 s | pvlib/NREL SPA, UTC; 0 = south, east negative |
| `Elevation[deg]` | degrees | solar elevation | recomputed by pipeline at 1 s; resampled from 1 s | apparent elevation, negative below the horizon |
| `Pac` | kW | modeled inverter AC power | computed at this resolution | King's model; values ≤ 0.001 set to zero |
| `Pdc` | kW | modeled array DC power | computed at this resolution | never averaged from a finer file |
| `TempModule` | °C | modeled module temperature | computed at this resolution | instantaneous model estimate |
| `TempCell` | °C | modeled cell temperature | computed at this resolution | instantaneous model estimate |
| `P_Solar_model_substituted` | boolean | substitution decision | computed at this resolution | `Pac >= 1.5 * P_Solar[kW]` |
| `P_Solar_clean[kW]` | kW | measured PV with model substitution | computed at this resolution | measured `P_Solar[kW]` remains present |
| `P_hybrid[kW]` | kW | clean PV plus measured wind power | computed at this resolution | `P_Solar_clean[kW] + P_Gaia[kW]` |

## Quality-flag columns

The pipeline writes one `int8` column `<column>_qc` per treated column in the 1-second files
(`WIND_DIR`, `Pressure`, `WIND_SPEED`, `HUMIDITY`, `TEMPERATURE`, `GHI`,
`POA Irr`, `P_Gaia`, `Azimuth`, `Elevation`). The physical expansion then
adds `P_Solar[kW]_qc`, `P_hybrid[kW]_qc`, and
`P_hybrid[kW]_qc_source` at each resolution. Exactly one code per value:

| Code | Name | Meaning |
|---|---|---|
| 0 | `QC_OK` | untouched |
| 1 | `QC_WRAPPED` | wind direction wrapped modulo 360 |
| 2 | `QC_PLACEHOLDER` | placeholder/stuck-sensor value replaced by NaN |
| 3 | `QC_DROPOUT_SHORT_FIXED` | wind-speed/humidity dropout ≤ 5 s, interpolated |
| 4 | `QC_DROPOUT_LONG_UNTREATED` | same dropout, longer run: value kept, flagged |
| 5 | `QC_GLITCH_SHORT_FIXED` | out-of-bounds value(s) ≤ 3 s, interpolated |
| 6 | `QC_MODEL_SUBSTITUTED` | `P_Solar_clean[kW]` uses modeled `Pac`; evaluated independently at this resolution |
| 7 | `QC_UNVERIFIED_PROVENANCE` | value unchanged and plausible, cause unknown (`P_Gaia` zeros) |
| 8 | `QC_RECOMPUTED` | value replaced by a model computation (`Azimuth`, `Elevation`) |
| 9 | `QC_GLITCH_LONG_UNTREATED_NAN` | out-of-bounds run too long to interpolate: set to NaN |
| 10 | `QC_ACTIVE_DAY` | `P_Gaia` on a day of confirmed real telemetry |

During pipeline resampling, each pipeline-owned `<column>_qc` is replaced by
`<column>_qc_worst` and `<column>_qc_frac_flagged` — see `METHODOLOGY.md`.
Code 6 and model-derived QC fields are not resampled; expansion recreates
them from that resolution's cleaned inputs. `P_hybrid[kW]_qc` inherits the
more severe of solar and wind under the canonical severity order, while
`P_hybrid[kW]_qc_source` records the winning constituent.
The reasoning behind every rule is in `CLEANING_DECISIONS.md`; the detection
rules are tabulated in `QC_SCHEMA.md`.

## Using the flags

```python
ok = df["Pressure[mbar]_qc"] == 0                       # untouched values only
gaia = df.loc[df["P_Gaia[kW]_qc"] == 10, "P_Gaia[kW]"]  # confirmed-real turbine data
clean_h = h[h["GHI[kW1m2]_qc_frac_flagged"] < 0.1]      # hourly buckets < 10 % flagged
```
