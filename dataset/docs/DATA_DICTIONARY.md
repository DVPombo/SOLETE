# SOLETE data dictionary (v4 files)

Site: SYSLAB, DTU Risø campus, Denmark — latitude 55.6867° N, longitude
12.0985° E, altitude 10 m. Co-located 11 kW Gaia wind turbine, 10 kW PV
system and a weather station.

## Files and time axis

| File | One row is | Columns |
|---|---|---|
| `SOLETE_Pombo_1sec_original_v4` | one second | 9 (measured only) |
| `SOLETE_Pombo_1sec_v4` | one second | 28 |
| `SOLETE_Pombo_1min_v4` | one minute | 36 |
| `SOLETE_Pombo_5min_v4` | five minutes | 36 |
| `SOLETE_Pombo_60min_v4` | one hour (`60min`; `1h` is accepted by the code as an input alias only) | 36 |

Each exists as `.h5` (in `data/hdf5/`) and `.parquet` (in `data/parquet/`, one more column: `timestamp`).

**Rows.** The record starts at 2018-06-01 00:00:00 UTC and covers 457 full days (457 x 86,400 = 39,484,800 seconds). The raw v3 files also hold the
**boundary second 2019-09-01 00:00:00** (the v3 hourly file has 10,969 rows, ending 2019-09-01 00:00:00; the raw 1 s file has 39,484,801 rows).
Nothing is removed, so the v4 files keep it, and the coarse files have one more (partial) bucket for it: with the boundary second
**39,484,801 / 658,081 / 131,617 / 10,969** rows (1 s / 1 min / 5 min / 60 min), without it 39,484,800 / 658,080 / 131,616 / 10,968. The
last bucket of each coarse file then contains a single second. The build derives the expected counts from the file's own span and checks
the grid is gap-free; the actual counts are in `manifest.json` and in `data/figshare_README.txt`.

- **Timestamps are UTC**, not local time (evidence: `CLEANING_DECISIONS.md` §2). There is no local-time column.
  In the `.h5` files the index is stored tz-naive; in the `.parquet` files it is the first column `timestamp`, typed `timestamp[ns, tz=UTC]` (the unit is pinned in code).
- Resampled rows are labelled by the **start** of their interval, `[T, T+period)`.
- Missing values are `NaN`. The 1-second grid itself has no gaps or duplicates.

### `SOLETE_Pombo_1sec_original_v4`: the raw data, sorted

The nine measured columns below, **nothing cleaned and nothing derived**, values bit-identical to the raw v3 file `SOLETE_Pombo_1sec.h5`. It
differs from the v3 file in exactly two ways: rows are sorted chronologically (v3 stores 457 daily blocks in shuffled order), and
`Azimuth[deg]` / `Elevation[deg]` are dropped because they are computed from the timestamp and the site, not measured (the original values
were faulty; the pipeline recomputes them, see below). Released so that the cleaning is transparent and reproducible from this file.

### `SOLETE_Pombo_1sec_v4` columns (28), in order

9 measured (cleaned, in `_original` order) + `Azimuth[deg]` + `Elevation[deg]` (recomputed) + 8 `<column>_qc` flags (`WIND_DIR`, `Pressure`,
`WIND_SPEED`, `HUMIDITY`, `TEMPERATURE`, `GHI`, `POA Irr`, `P_Gaia`) + the 9 model columns below. The coarser files have the same 9 measured
columns and the 2 angles (resampled), then `_qc_worst` and `_qc_frac_flagged` for each of the 8 flags, then the 9 model columns (36).
The fact that Azimuth/Elevation are recomputed is stated here and in the Parquet metadata; there is no flag column for it (it would be a
constant 8 on every row, `QC_SCHEMA.md` §3b).

## Measurement columns

The **Provenance** column says how each value was obtained, in each file family:
*measured* (as recorded; in `_original` unchanged, in the 1 s file after the cleaning rules listed), *resampled from 1 s* (aggregated from the cleaned
1-second data, see `METHODOLOGY.md`), *recomputed by the pipeline* (pvlib, per timestamp of the 1 s file),
*computed at this resolution* (model-derived: calculated from that file's own cleaned inputs by
`solete/expansion.py`, never averaged from a finer resolution). The per-column provenance is also stored in the Parquet metadata
(`release_meta.column_provenance`).

| Column | Unit | Meaning | Provenance (`_original` / 1 s / coarser) | Things to know |
|---|---|---|---|---|
| `TEMPERATURE[degC]` | °C | ambient air temperature | measured / measured, cleaned / resampled from 1 s | one −40.1 °C second removed as a glitch |
| `HUMIDITY[%]` | **fraction 0–1** | relative humidity | measured / measured, cleaned / resampled from 1 s | despite `%` in the name the values are fractions; 2018-11-17 is NaN (stuck sensor, 140–200 %) |
| `WIND_SPEED[m1s]` | m/s | wind speed | measured / measured, cleaned / resampled from 1 s | joint zero-dropouts with humidity interpolated if ≤ 5 s |
| `WIND_DIR[deg]` | degrees | wind direction | measured / measured, cleaned / resampled from 1 s | wrapped to [0, 360); resampled with a circular mean |
| `GHI[kW1m2]` | kW/m² | global horizontal irradiance | measured / measured, cleaned / resampled from 1 s | bound-checked to [0, 1.5] |
| `POA Irr[kW1m2]` | kW/m² | plane-of-array irradiance | measured / measured, cleaned / resampled from 1 s | bound-checked to [0, 1.6] |
| `Pressure[mbar]` | mbar (= hPa) | atmospheric pressure | measured / measured, cleaned / resampled from 1 s | **valid on 2019-01-16 only**; every other second was a placeholder and is NaN |
| `P_Gaia[kW]` | kW | active power, 11 kW Gaia wind turbine | measured / measured (values untouched) / resampled from 1 s | **only 2018-08-31 and 2019-05-25 are confirmed real telemetry**; elsewhere the recorded zeros may mean "turbine off" or "channel not logged" (unresolved) |
| `P_Solar[kW]` | kW | active power, 10 kW PV inverter | measured / measured, never overwritten / resampled from 1 s | no cleaning rule applied, no pipeline flag column. In the expanded files `P_Solar[kW]_qc` (code 6 only) and `P_Solar_clean[kW]` are added |
| `Azimuth[deg]` | degrees | solar azimuth | **absent** / recomputed by the pipeline / resampled from 1 s\* | **computed, not measured** (pvlib/NREL SPA, UTC, site constants in `data/README.md`). 0 = south, east negative, range [-180, 180) |
| `Elevation[deg]` | degrees | solar elevation | **absent** / recomputed by the pipeline / resampled from 1 s\* | **computed, not measured.** Apparent (refraction-corrected), negative below the horizon (not clipped to 0 at night as in v3) |

\* The 60-minute/5-minute/1-minute `Azimuth`/`Elevation` are the mean of the 1-second pvlib values (a **circular** mean for the azimuth, so the
bucket around solar midnight, where the 1 s values jump between +180 and −180, is correct; a plain mean for the elevation), not pvlib evaluated at
the bucket timestamp.

## Model-derived columns (expanded / "full" files)

Added by `solete.expansion.expand_physical` from the same file's cleaned inputs
(`POA Irr`, `TEMPERATURE`, `WIND_SPEED`, `P_Solar`, `P_Gaia`). **Computed at each resolution**; never averaged from
a finer one (a 1-second value is an instantaneous model estimate — module thermal inertia is minutes — and the 1-second
substitution flag is not comparable with the hourly one). All are row-wise: no window, no neighbour.

| Column | Unit | Provenance | Definition |
|---|---|---|---|
| `Pac` | kW | computed at this resolution | King's PV model AC power of both inverter channels; `<= 0.001` set to 0; NaN input gives 0 |
| `Pdc` | kW | computed at this resolution | same model, DC array power |
| `TempModule`, `TempCell` | °C | computed at this resolution | module / cell temperature (mean of the two strings) |
| `P_Solar[kW]_qc` | int8 | computed at this resolution | code 6 (`QC_MODEL_SUBSTITUTED`) where `Pac >= 1.5 * P_Solar[kW]` (unrounded `Pac`) **and** stored `Pac > 0`; NaN measurement → not flagged; else 0. This is the only record of the substitution (there is no separate boolean column) |
| `P_Solar_clean[kW]` | kW | computed at this resolution | `P_Solar[kW]`, replaced by the unrounded `Pac` where substituted, then values `<= 0.001` set to 0. The platform's working `P_Solar[kW]` |
| `P_hybrid[kW]` | kW | computed at this resolution | `P_Solar_clean[kW] + P_Gaia[kW]` |
| `P_hybrid[kW]_qc` | int8 | computed at this resolution | the more severe of `P_Solar[kW]_qc` and `P_Gaia[kW]_qc` (`P_Gaia[kW]_qc_worst` in the resampled files), ties to `P_Solar` (`QC_SCHEMA.md` §8) |
| `P_hybrid[kW]_qc_source` | int8 | computed at this resolution | which constituent the hybrid flag was taken from: `0` none (hybrid flag is 0), `1` `P_Solar[kW]` (ties too), `2` `P_Gaia[kW]`. Labels: `solete.qc_codes.SOURCE_LABELS`; table in `QC_SCHEMA.md` §8 |

`TempModule_RP` (Rincón–Pombo thermodynamic model) is not part of the files: it carries the previous row's temperature, so it is
not row-wise and depends on the time step; it stays an on-demand platform feature.

## Quality-flag columns

One `int8` column `<column>_qc` per treated column, in the 1-second file
(`WIND_DIR`, `Pressure`, `WIND_SPEED`, `HUMIDITY`, `TEMPERATURE`, `GHI`,
`POA Irr`, `P_Gaia`: **8 columns**; not `P_Solar` and not `Azimuth`/`Elevation`). Exactly one
code per value:

| Code | Name | Meaning |
|---|---|---|
| 0 | `QC_OK` | untouched |
| 1 | `QC_WRAPPED` | wind direction wrapped modulo 360 |
| 2 | `QC_PLACEHOLDER` | placeholder/stuck-sensor value replaced by NaN |
| 3 | `QC_DROPOUT_SHORT_FIXED` | wind-speed/humidity dropout ≤ 5 s, interpolated |
| 4 | `QC_DROPOUT_LONG_UNTREATED` | same dropout, longer run: value kept, flagged |
| 5 | `QC_GLITCH_SHORT_FIXED` | out-of-bounds value(s) ≤ 3 s, interpolated |
| 6 | `QC_MODEL_SUBSTITUTED` | **platform-owned**: measured `P_Solar` is at least 1.5 × below the PV model (`P_Solar[kW]_qc` of expanded files only; computed per resolution; never emitted by the pipeline) |
| 7 | `QC_UNVERIFIED_PROVENANCE` | value unchanged and plausible, cause unknown (`P_Gaia` zeros) |
| 8 | `QC_RECOMPUTED` | **reserved, not emitted by any v4 column** (the two constant-8 Azimuth/Elevation flag columns were dropped, decision D1); the number stays defined so earlier development files load, and is never reused |
| 9 | `QC_GLITCH_LONG_UNTREATED_NAN` | out-of-bounds run too long to interpolate: set to NaN |
| 10 | `QC_ACTIVE_DAY` | `P_Gaia` on a day of confirmed real telemetry |
| 11 | `QC_UNTREATED_IMPLAUSIBLE` | legacy v3 files only (platform raw-value checks); never in v4 |

In the 1-minute, 5-minute and 60-minute files each `<column>_qc` is replaced by
`<column>_qc_worst` and `<column>_qc_frac_flagged` — see `METHODOLOGY.md`.
The reasoning behind every rule is in `CLEANING_DECISIONS.md`; the detection
rules are tabulated in `QC_SCHEMA.md`.

## Using the flags

```python
ok = df["Pressure[mbar]_qc"] == 0                       # untouched values only
gaia = df.loc[df["P_Gaia[kW]_qc"] == 10, "P_Gaia[kW]"]  # confirmed-real turbine data
clean_h = h[h["GHI[kW1m2]_qc_frac_flagged"] < 0.1]      # hourly buckets (SOLETE_Pombo_60min_v4) < 10 % flagged
```
