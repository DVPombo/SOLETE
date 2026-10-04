# SOLETE data dictionary (v4, cleaned files)

Site: SYSLAB, DTU Risø campus, Denmark — latitude 55.6867° N, longitude
12.0985° E, altitude 10 m. Co-located 11 kW Gaia wind turbine, 10 kW PV
system and a weather station.

## Files and time axis

| File family | Rows (expected) | One row is |
|---|---|---|
| `*_1sec` | 39,484,800 | one second; 457 complete days, **2018-06-01 00:00:00 → 2019-08-31 23:59:59** |
| `*_1min` | 658,080 | one minute |
| `*_5min` | 131,616 | five minutes |
| `*_1h` | 10,968 | one hour (called `60min` in v3) |

- **Timestamps are UTC**, not local time (evidence: `CLEANING_DECISIONS.md` §2).
  In the `.h5` files the index is stored tz-naive; in the `.parquet` files it is
  the column `timestamp`, typed `timestamp[ns, UTC]`.
- Resampled rows are labelled by the **start** of their interval, `[T, T+period)`.
- Missing values are `NaN`. The 1-second grid itself has no gaps or duplicates.

## Measurement columns

| Column | Unit | Meaning | Things to know |
|---|---|---|---|
| `TEMPERATURE[degC]` | °C | ambient air temperature | one −40.1 °C second removed as a glitch |
| `HUMIDITY[%]` | **fraction 0–1** | relative humidity | despite `%` in the name the values are fractions; 2018-11-17 is NaN (stuck sensor, 140–200 %) |
| `WIND_SPEED[m1s]` | m/s | wind speed | joint zero-dropouts with humidity interpolated if ≤ 5 s |
| `WIND_DIR[deg]` | degrees | wind direction | wrapped to [0, 360); resampled with a circular mean |
| `GHI[kW1m2]` | kW/m² | global horizontal irradiance | bound-checked to [0, 1.5] |
| `POA Irr[kW1m2]` | kW/m² | plane-of-array irradiance | bound-checked to [0, 1.6] |
| `Pressure[mbar]` | mbar (= hPa) | atmospheric pressure | **valid on 2019-01-16 only**; every other second was a placeholder and is NaN |
| `P_Gaia[kW]` | kW | active power, 11 kW Gaia wind turbine | values untouched; **only 2018-08-31 and 2019-05-25 are confirmed real telemetry**; elsewhere the recorded zeros may mean "turbine off" or "channel not logged" (unresolved) |
| `P_Solar[kW]` | kW | active power, 10 kW PV inverter | no cleaning rule applied, no flag column |
| `Azimuth[deg]` | degrees | solar azimuth | **computed, not measured** (pvlib/NREL SPA, UTC). 0 = south, east negative |
| `Elevation[deg]` | degrees | solar elevation | **computed, not measured.** Apparent (refraction-corrected), negative below the horizon (not clipped to 0 at night as in v3) |

## Quality-flag columns

One `int8` column `<column>_qc` per treated column, in the 1-second files
(`WIND_DIR`, `Pressure`, `WIND_SPEED`, `HUMIDITY`, `TEMPERATURE`, `GHI`,
`POA Irr`, `P_Gaia`, `Azimuth`, `Elevation`; **not** `P_Solar`). Exactly one
code per value:

| Code | Name | Meaning |
|---|---|---|
| 0 | `QC_OK` | untouched |
| 1 | `QC_WRAPPED` | wind direction wrapped modulo 360 |
| 2 | `QC_PLACEHOLDER` | placeholder/stuck-sensor value replaced by NaN |
| 3 | `QC_DROPOUT_SHORT_FIXED` | wind-speed/humidity dropout ≤ 5 s, interpolated |
| 4 | `QC_DROPOUT_LONG_UNTREATED` | same dropout, longer run: value kept, flagged |
| 5 | `QC_GLITCH_SHORT_FIXED` | out-of-bounds value(s) ≤ 3 s, interpolated |
| 6 | *(reserved)* | belongs to the SOLETE forecasting platform; never emitted here |
| 7 | `QC_UNVERIFIED_PROVENANCE` | value unchanged and plausible, cause unknown (`P_Gaia` zeros) |
| 8 | `QC_RECOMPUTED` | value replaced by a model computation (`Azimuth`, `Elevation`) |
| 9 | `QC_GLITCH_LONG_UNTREATED_NAN` | out-of-bounds run too long to interpolate: set to NaN |
| 10 | `QC_ACTIVE_DAY` | `P_Gaia` on a day of confirmed real telemetry |

In the 1-minute, 5-minute and 1-hour files each `<column>_qc` is replaced by
`<column>_qc_worst` and `<column>_qc_frac_flagged` — see `METHODOLOGY.md`.
The reasoning behind every rule is in `CLEANING_DECISIONS.md`; the detection
rules are tabulated in `QC_SCHEMA.md`.

## Using the flags

```python
ok = df["Pressure[mbar]_qc"] == 0                       # untouched values only
gaia = df.loc[df["P_Gaia[kW]_qc"] == 10, "P_Gaia[kW]"]  # confirmed-real turbine data
clean_h = h[h["GHI[kW1m2]_qc_frac_flagged"] < 0.1]      # hourly buckets < 10 % flagged
```
