# SOLETE Data Dictionary

This document describes every column found in the real SOLETE `.h5` files shipped at the
repo root, as delivered (i.e. **before** `ExpandSOLETE()` adds derived columns), plus the
one derived column (`P_Solar_model_substituted`) that Phase 0.5 made a real, first-class
column instead of a silent substitution.

It was built by running `scripts/inspect_dataset.py` against the two real files present
in this repo:

- `SOLETE_short.h5` — 24 rows, 1-hour resolution, 2018-06-01 00:00 → 23:00 (one day).
- `SOLETE_Pombo_60min.h5` — 10,969 rows, 1-hour resolution, 2018-06-01 → 2019-09-01
  (~15 months, matches the 15-month span reported in the Data in Brief paper [1]).

Both files are **1-hour resolution**. No 1-second, 1-minute, or 5-minute file was found
at the repo root — see the "Resolutions found" note at the bottom of this file, and
`RESOLUTIONS.md` for what this means for the `Control_Var['resolution']` options the
code exposes.

Where a fact comes only from the paper (sensor make/model, exact site, calibration) and
not from anything in the files or code, it's marked **(paper-sourced)**. Everything else
was checked directly against the two files.

---

## Source and general provenance

Recorded at SYSLAB, a distributed-energy-resources test laboratory at DTU Wind and
Energy Systems, Denmark, using a meteorological station, an 11 kW Gaia wind turbine, and
a 10 kW-class PV array/inverter, transferred to a central server **(paper-sourced;**
Pombo, Gehrke & Bindner 2022, *Data in Brief* 42, 108046). Exact sensor make/model,
calibration records, and precise GPS coordinates are not in the repo files or code —
<!-- TODO: confirm with maintainer --> if this level of detail is wanted, it should come
from the maintainer or the full paper text, not be guessed here.

---

## Column-by-column

### `TEMPERATURE[degC]`
- **Human-readable name:** Ambient (dry-bulb) air temperature
- **Units:** Degrees Celsius
- **Source:** Meteorological station at SYSLAB (paper-sourced for instrument detail)
- **Resolution(s) observed:** short, 60min
- **Observed range:** short: 13.05 – 27.26 °C. 60min: **-4.40 – 44.24 °C** (365 of 10,969
  rows below 0 °C, all in Danish winter months — expected).
- **Caveats:** The 60min max of 44.24 °C (2018-07-31) and several other 40+ °C readings
  cluster in July–August 2018, the summer of a well-documented NW-European heatwave, so
  this is plausibly real rather than a sensor fault, but 44 °C is above Denmark's
  national temperature record and the station may be sited near reflective/heat-emitting
  equipment (PV array, inverter housing) that could bias it warm. `<!-- TODO: confirm with maintainer -->`
  whether these summer-2018 highs are known-good or a station siting artifact.

### `HUMIDITY[%]`
- **Human-readable name:** Relative humidity
- **Units:** Despite the `[%]` in the column name, this is stored as a **fraction on
  [0, 1]**, not a percentage on [0, 100] — confirmed both by the observed value range
  (short: 0.34–0.88; 60min bulk of data in 0–1) and by `Functions.py`'s
  `Rincon_Pombo_ThermodynamicModel`, which clips `humidity[i] > 1` down to `1.0` before
  feeding it to `CoolProp.HAPropsSI` as a relative-humidity fraction argument `'R'`.
- **Source:** Meteorological station at SYSLAB (paper-sourced for instrument detail)
- **Resolution(s) observed:** short, 60min
- **Observed range:** short: 0.34 – 0.88 (physically valid throughout). 60min: **0.00 –
  2.70**, with **188 of 10,969 rows (1.7%) above 1.0** — physically impossible for a
  fraction-based relative humidity.
- **Caveats:** The >1.0 rows are a genuine data-quality issue on the 60min file, not
  present (at this sample size) in the short file. See `KNOWN_ISSUES.md`. The code's own
  clip-to-1.0 logic only fixes this locally inside one derived-temperature calculation;
  it does **not** fix the underlying `HUMIDITY[%]` column, so any other use of this
  column downstream still sees the invalid values.

### `WIND_SPEED[m1s]`
- **Human-readable name:** Wind speed
- **Units:** Metres per second (`m1s` in the column name is a filesystem/HDF5-safe
  encoding of `m/s`)
- **Source:** Meteorological station at SYSLAB (paper-sourced for instrument detail)
- **Resolution(s) observed:** short, 60min
- **Observed range:** short: 0.61 – 2.21 m/s. 60min: 0.00 – 23.15 m/s.
- **Caveats:** None found. The 60min range spans the wind turbine's cut-in (3.5 m/s) to
  near/above cut-out (25 m/s) as documented in the `WT` dict in `Functions.py`, which is
  consistent with a real, variable wind record.

### `WIND_DIR[deg]`
- **Human-readable name:** Wind direction
- **Units:** Degrees, meteorological convention presumed (paper-sourced convention;
  not stated in-file or in-code)
- **Source:** Meteorological station at SYSLAB (paper-sourced for instrument detail)
- **Resolution(s) observed:** short, 60min
- **Observed range:** short: 68.05 – 304.89° (valid). 60min: **0.00 – 639.34°**, with
  **103 of 10,969 rows (0.94%) above 360°**, up to 639.34°.
- **Caveats:** Values above 360° are not a valid compass bearing. This strongly suggests
  these rows are themselves the output of an **arithmetic (non-circular) mean** applied
  during aggregation to this file's resolution — averaging e.g. 350° and 10° gives 180°
  under a naive mean (wrong) but a value like 639° can only arise from summing/averaging
  angles without wrapping them back into [0, 360) afterward. See `RESOLUTIONS.md` for the
  fuller aggregation-methodology discussion; this is filed as an open, unverified
  root-cause finding in `KNOWN_ISSUES.md`. `<!-- TODO: confirm with maintainer -->`

### `GHI[kW1m2]`
- **Human-readable name:** Global Horizontal Irradiance
- **Units:** kW/m² (`kW1m2` encodes `kW/m2`)
- **Source:** Meteorological station pyranometer at SYSLAB (paper-sourced for instrument
  detail)
- **Resolution(s) observed:** short, 60min
- **Observed range:** short: 0.00 – 0.639 kW/m². 60min: 0.00 – 0.910 kW/m².
- **Caveats:** No negative values in either file (irradiance can't be physically
  negative); ranges are within the plausible envelope for GHI at this latitude
  (theoretical max ~1.0–1.1 kW/m² at solar noon in clear conditions).

### `POA Irr[kW1m2]`
- **Human-readable name:** Plane-of-Array (POA) irradiance — irradiance measured in the
  plane of the tilted PV array, i.e. what the panels themselves actually receive
- **Units:** kW/m²
- **Source:** Pyranometer mounted in the PV array plane at SYSLAB (paper-sourced for
  instrument detail). This is the column `PV_Performance_Model()` expects by default
  (`colirra='POA Irr[kW1m2]'`) — the function's own docstring explicitly warns "make sure
  you are feeding Epoa and not GHI", i.e. `GHI[kW1m2]` and `POA Irr[kW1m2]` are not
  interchangeable inputs to that model.
- **Resolution(s) observed:** short, 60min
- **Observed range:** short: 0.00 – 0.855 kW/m². 60min: 0.00 – 0.962 kW/m².
- **Caveats:** No negative values; POA can legitimately exceed GHI (tilt + albedo can add
  to direct beam contribution near solar noon in some geometries), and the max values
  above are consistent with that (POA max > GHI max in both files).

### `P_Gaia[kW]`
- **Human-readable name:** Active power output of the Gaia wind turbine
- **Units:** kW
- **Source:** 11 kW Gaia wind turbine at SYSLAB (paper-sourced for turbine identity;
  nameplate rating `Pn: 11` kW is in the `WT` dict in `Functions.py`)
- **Resolution(s) observed:** short, 60min
- **Observed range:** short: 0.00 – 0.246 kW (low — the short sample is a single low-wind
  day). 60min: 0.00 – 10.18 kW, consistent with the 11 kW nameplate rating.
- **Caveats:** No negative values found; no values exceed nameplate rating.

### `P_Solar[kW]`
- **Human-readable name:** Measured active power output of the PV array/inverter
- **Units:** kW
- **Source:** PV inverter at SYSLAB (paper-sourced for array identity; combined nameplate
  from the `PV` dict in `Functions.py`, `Pmp_stc=[165, 125] W × Ns × Np` per string, sums
  to ~7.44 kW DC)
- **Resolution(s) observed:** short, 60min
- **Observed range:** short: 0.00 – 5.41 kW. 60min: 0.00 – 6.98 kW (under the ~7.44 kW DC
  nameplate, as expected once inverter/thermal derating is accounted for).
- **Caveats:** **This column is not the raw sensor reading everywhere.** During
  `ExpandSOLETE()`, any row where King's PV performance model estimate (`Pac`) is ≥ 1.5×
  the measured value gets its `P_Solar[kW]` **replaced** by the modeled `Pac` (this was a
  silent substitution before Phase 0.5; it's now flagged by `P_Solar_model_substituted`,
  see below). Values ≤ 0.001 are floored to exactly 0 afterward ("smoothing zeros" in
  `ExpandSOLETE`). Anyone using `P_Solar[kW]` from the **expanded** dataset should treat
  it as "measured, except where flagged as model-substituted" rather than as a pure
  sensor column. The **raw, as-delivered** `P_Solar[kW]` in the two files inspected here
  (before `ExpandSOLETE` runs) is the true sensor reading.

### `Pressure[mbar]`
- **Human-readable name:** Atmospheric pressure
- **Units:** Millibars (confirmed by `Rincon_Pombo_ThermodynamicModel`, which does
  `p = data['Pressure[mbar]'] * 100` to convert to Pascals for `CoolProp`)
- **Source:** Meteorological station at SYSLAB (paper-sourced for instrument detail)
- **Resolution(s) observed:** short, 60min
- **Observed range:** short: 1013.26 – 1017.47 mbar (fully realistic sea-level-ish
  pressure). 60min: **992.37 – 3000.00 mbar**.
- **Caveats:** **Serious data-quality issue on the 60min file, not present on the short
  file.** 10,477 of 10,969 rows (95.5%) are pegged at **exactly 1000.000000 mbar**, 477
  rows (4.3%) at exactly 2000.0, and 2 rows at exactly 3000.0. Only ~13 rows carry a
  non-round, plausible pressure value (992–998 mbar). This looks like a placeholder /
  fill value pattern (1000, 2000, 3000 are suspiciously round and far too clustered to be
  real atmospheric readings), not real pressure measurements for the vast majority of the
  60min file. See `KNOWN_ISSUES.md`. `<!-- TODO: confirm with maintainer -->`

### `Azimuth[deg]` *(60min file only)*
- **Human-readable name:** Solar azimuth angle
- **Units:** Degrees
- **Source:** Not documented anywhere — not referenced by name in `Functions.py`,
  `RunMe.py`, or `MLForecasting.py`, and not listed in `Control_Var['PossibleFeatures']`
  in `MLForecasting.py`. Presumably computed from a solar-position calculation at
  dataset-build time, but that computation isn't in this repo. `<!-- TODO: confirm with maintainer -->`
- **Resolution(s) observed:** 60min only. **Not present in `SOLETE_short.h5`.**
- **Observed range:** -56.91 – 44.79°, but **10,959 of 10,969 rows (99.9%) are exactly
  0.0**; only 10 rows, all on a single calendar day (2019-01-16), carry a non-zero value.
- **Caveats:** This does not look like a usable solar-azimuth time series — it's
  essentially constant-zero with one day of real-looking values. See `KNOWN_ISSUES.md`.
  Treat as unreliable/unpopulated until confirmed otherwise.

### `Elevation[deg]` *(60min file only)*
- **Human-readable name:** Solar elevation angle
- **Units:** Degrees
- **Source:** Same as `Azimuth[deg]` — undocumented, unreferenced in code.
  `<!-- TODO: confirm with maintainer -->`
- **Resolution(s) observed:** 60min only. **Not present in `SOLETE_short.h5`.**
- **Observed range:** 0.00 – 12.88°, with **10,960 of 10,969 rows (99.9%) exactly 0.0**;
  the only non-zero values are the same single day, 2019-01-16, as `Azimuth[deg]`, and
  even that day's peak elevation (12.88° at solar noon in mid-January) is a physically
  plausible value for Denmark's latitude in winter — but the surrounding ~15 months are
  all zero, including summer days where solar elevation should be much higher.
- **Caveats:** Same as `Azimuth[deg]` — effectively unpopulated outside one day. See
  `KNOWN_ISSUES.md`.

### `P_Solar_model_substituted` *(derived column, added by `ExpandSOLETE()`, not present in the raw files)*
- **Human-readable name:** Flag: was this row's `P_Solar[kW]` replaced by the physics
  model estimate?
- **Units:** Boolean
- **Source:** Computed in `Functions.py::ExpandSOLETE()` as
  `data['Pac'] >= 1.5*data['P_Solar[kW]']`, where `Pac` is King's PV performance model
  output (see `PV_Performance_Model()`). This is a Phase 0.5 fix: the substitution
  already happened in earlier versions of the code, silently; this column makes it
  traceable.
- **Resolution(s) it appears in:** Any resolution, once `ExpandSOLETE()` has run — it is
  **not** a column in the raw `.h5` files, only in the in-memory/expanded DataFrame (and
  in a saved `*_Expanded.h5` file if `Control_Var['SOLETE_save']=True`).
- **Meaning:** `True` means the row's measured `P_Solar[kW]` looked implausible relative
  to the physics-model estimate (the measured value was less than ~2/3 of modeled `Pac`,
  i.e. modeled ≥ 1.5× measured) and was overwritten with the modeled value as a
  noise/curtailment cleaning step. `False` means the original sensor reading was kept.
- **Real substitution rate (computed from both real files, this phase):**
  - `SOLETE_short.h5`: **20.83%** of rows (5 of 24)
  - `SOLETE_Pombo_60min.h5`: **38.33%** of rows (4,204 of 10,969)
  See `KNOWN_ISSUES.md` for discussion of why these two rates differ so much and what
  that might imply.

---

## Column-set comparison between the two real files

| Column | `SOLETE_short.h5` | `SOLETE_Pombo_60min.h5` |
|---|---|---|
| `TEMPERATURE[degC]` | ✅ | ✅ |
| `HUMIDITY[%]` | ✅ | ✅ |
| `WIND_SPEED[m1s]` | ✅ | ✅ |
| `WIND_DIR[deg]` | ✅ | ✅ |
| `GHI[kW1m2]` | ✅ | ✅ |
| `POA Irr[kW1m2]` | ✅ | ✅ |
| `P_Gaia[kW]` | ✅ | ✅ |
| `P_Solar[kW]` | ✅ | ✅ |
| `Pressure[mbar]` | ✅ | ✅ |
| `Azimuth[deg]` | ❌ | ✅ |
| `Elevation[deg]` | ❌ | ✅ |

The two extra columns in the 1-hour full dataset are not anticipated by the README, the
paper's abstract/keywords, or `Control_Var['PossibleFeatures']` in `MLForecasting.py` —
and as documented above, they're ~99.9% constant zero, so their presence looks more like
a leftover/partial computation than an intentional, documented addition. This is called
out explicitly rather than silently merged into "the" SOLETE column list.

No columns are present in the short file but absent from the 60min file — the short
sample is a strict subset of the 60min file's columns.

## Cross-check against paper/README-implied variable set

The paper abstract and README keywords describe: timestamp, air temperature, relative
humidity, pressure, wind speed, wind direction, GHI, POA irradiance, and active power
from both the wind turbine and PV inverter. Every one of those is present in both real
files under the column names above. `Azimuth[deg]` and `Elevation[deg]` (60min file
only) are the only columns found that are **not** anticipated by the paper/README — see
above.

## Resolutions found on disk

Only **1-hour-resolution** files (`SOLETE_short.h5`, `SOLETE_Pombo_60min.h5`) are present
at the repo root. `Control_Var['resolution']` in `MLForecasting.py`/`Functions.py`
accepts `'1sec'`, `'1min'`, `'5min'`, or `'60min'`, implying `SOLETE_Pombo_1sec.h5`,
`SOLETE_Pombo_1min.h5`, and `SOLETE_Pombo_5min.h5` should also exist as companion files
(the README explicitly discusses the 1-second file's size), but none of those three are
in this repo. Everything in this document is therefore verified only for the 1-hour
resolution; claims about the other three resolutions' column sets or value ranges would
be extrapolation, not verification, so none are made here.

---

## References
[1] Pombo, D. V., Gehrke, O., & Bindner, H. W. (2022). SOLETE, a 15-month long holistic
    dataset including: Meteorology, co-located wind and solar PV power from Denmark with
    various resolutions. *Data in Brief*, 42, 108046.
