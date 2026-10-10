# SOLETE 1-second file: cleaning decisions

This documents what `clean_solete_1sec.py` changed, why, and what is still
open. It ships alongside the cleaned files so nobody has to re-derive any of this
from the scripts. Terms: *raw* = the 1-second file as published in v3;
*v3* = the originally published release.

## 1. Row order
The raw file stores 457 daily blocks in shuffled order (confirmed:
`followup_A_index_order` — a complete, gap-free, duplicate-free 1-second grid,
just not written in time order). The cleaned file is sorted chronologically.
Nothing is added or removed.

## 2. Timestamps are UTC, not local time
No `DATA_DICTIONARY.md` statement about this was found. The GHI-weighted
daily centroid time of day matches computed solar noon in UTC in every month
of the record, including summer — ruling out both CET (UTC+1) and CEST
(UTC+2). `solar_position.py` now defaults to UTC accordingly.
The one populated day in the *original* Azimuth/Elevation columns
(2019-01-16) instead best fits UTC+1 — evidence that whoever computed those
values assumed local time, one hour off from what the sensors actually
recorded.

## 3. WIND_DIR[deg]: wrapped to [0, 360)
Values above 360° (1,153 rows, 362°–712°, almost all on 2018-11-17) are extra
sensor revolutions, not different bearings (maintainer's call). Fixed with
`value mod 360`. This also fixes the 103 hourly values above 360° in the
*original* published hourly file, which turn out not to come from this raw
file at all — see `followup_B_wind_dir`: those hourly values exceed the raw
hour's own max, so they must trace to an earlier/different raw version.
Re-running `resample_solete.py` on the cleaned file regenerates a correct
hourly WIND_DIR with the existing circular-mean logic (still needed after the
wrap — averaging across the 359°/0° boundary is still wrong as a plain mean).

## 4. Pressure[mbar]: placeholder values → NaN
A sample is treated as a placeholder if it's an exact multiple of 1000 (any
count — 1000.0 is 99.9% of the raw column, 2000.0 a smaller block, and this
also catches isolated 3000.0 glitches too rare to register as a "run"), OR
if it sits in a run of ≥300 consecutive bit-for-bit identical samples,
regardless of whether that value is round (real pressure drifts second to
second; a flatline for minutes at a time is itself evidence of a stuck
sensor). The first real run caught only the round-number case and left three
"real" days instead of one — the extra two turned out to be hour-plus
plateaus at non-round values (997.5, 997.6, 997.4 mbar) that the
round-number-only test missed. The flatline test added afterward catches
these too. Only 2019-01-16 has genuinely varying pressure (~991–1005 mbar).

**Open question:** could a real but brief (<300s) excursion to a round
number, or a short non-round plateau, still slip through? Both thresholds
(exact-multiple-of-1000, flatline run ≥300s) are judgment calls — check
`pressure_n_flagged_exact_multiples_of_1000` /
`pressure_n_flagged_flatline_only` / `pressure_flatline_only_example_values`
in the run's report before trusting it.

## 5. WIND_SPEED[m1s] & HUMIDITY[%]: joint-zero dropout
A second where **both** wind speed and humidity read exactly 0 is treated as
a sensor dropout, not real calm-and-bone-dry air (0% relative humidity is not
physically plausible at this site). Runs of 5 seconds or fewer are linearly
interpolated between the valid values just before/after the run (or
persisted from one side if the run touches a boundary). Longer runs are left
as recorded and flagged for manual review — the threshold is a judgment call,
not a hard physical fact, so it's a `--dropout-max-run` CLI argument if it
needs revisiting.

## 6. Generic "logger glitch" pass
Any single value outside a per-column physical bound is treated the same
way as #5: short runs (≤3 seconds by default) are interpolated, longer runs
are left alone and flagged. Bounds used (all **provisional** — not validated
against site records, just against what showed up in the profiler):

| Column | Bound | Rationale |
|---|---|---|
| TEMPERATURE[degC] | [-25, 40] | only known violation: the single -40.1°C second |
| HUMIDITY[%] | ≤ 1.0 | stored as a 0–1 fraction; lower bound handled by #5 instead |
| Pressure[mbar] | [900, 1100] | checked after sentinels are removed |
| GHI[kW1m2] | [0, 1.5] | |
| POA Irr[kW1m2] | [0, 1.6] | |

The known ~1-day HUMIDITY anomaly on 2018-11-17 (constant 1.4–2.0, i.e.
140–200%) is **long enough that this pass leaves it too long to interpolate**.
Unlike the WS/HUM dropout pass (#5), where an untreated run keeps its
recorded value because that value is still individually plausible (e.g. a
genuine 0), a value outside these physical bounds is impossible by
construction — so an untreated run here is **overwritten with NaN** instead
of left in place, and flagged `QC_GLITCH_LONG_UNTREATED_NAN` for a human
decision. It looks like a miscalibration for that whole day, not a transient
glitch, and linearly interpolating across a full day would manufacture data,
not clean it; leaving 140–200% humidity sitting in the file would be worse
than admitting the day is missing.

WIND_SPEED has no generic upper bound: the raw max (29 m/s) is a plausible
storm gust, so nothing there is currently flagged as a glitch.

## 7. P_Gaia[kW]: values kept as recorded; cause of the zeros is undocumented on purpose
**No values are changed.** Every row is flagged instead:
- `QC_ACTIVE_DAY` on 2018-08-31 and 2019-05-25 — the only two days with
  telemetry that behaves like a real turbine (power tracks wind speed on a
  smooth curve consistent with the Gaia-Wind 133-11kW datasheet).
- `QC_UNVERIFIED_PROVENANCE` everywhere else (99.56% of rows) — recorded as
  0, but **it is not known** whether this means the turbine was genuinely
  offline or the channel simply wasn't being logged. Evidence for "not
  logged" rather than "offline": the switch to/from real values happens
  exactly at midnight boundaries, and Pressure/Azimuth/Elevation show the
  identical pattern of being populated on isolated days and
  placeholder/zero elsewhere. This is circumstantial, not confirmed — only
  DTU SYSLAB's own operations or logging records could settle it. This
  **substantially narrows but does not fully confirm** the cause; don't
  treat it as established.

The SOLETE forecasting platform inherits `P_Gaia[kW]_qc` for its derived
hybrid-power column automatically; no platform code change is needed.

**Validation note:** comparing the recomputed hourly Azimuth/Elevation
against the *original* published hourly file will show ~0% agreement and a
large mean difference. That's expected, not a bug — the original is ~99.9%
zero (a sparse placeholder), while the recomputed columns are continuous and
populated for every hour. They're supposed to look nothing alike.

## 8. Azimuth[deg] / Elevation[deg]: recomputed from scratch
The published values are dropped entirely (they carry a still-unexplained
constant offset from pvlib of about -4.72° in azimuth and -0.45° in
elevation, on top of the timezone error in #2 — see
`solar_position_validation_v2`; the size of that offset is not pursued
further, see "Reconciliation items" below). **None of the original values
are delivered:** both columns are 100% pvlib-recomputed, no original values
mixed in. Site: 55.6867°N, 12.0985°E (Risø, Denmark), 10 m altitude
(`solar_position.py`: `SITE_LATITUDE`, `SITE_LONGITUDE`, `SITE_ALTITUDE_M`).
Replaced with a full pvlib (NREL SPA) computation for every row:
- **Timestamps:** UTC (see #2).
- **Azimuth convention:** south-referenced, east negative (0=south), to
  match the original column's apparent convention — not pvlib's own
  default (0=north, clockwise).
- **Elevation:** *apparent* (refraction-corrected), **not clipped at
  night** — it goes negative below the horizon. The published file instead
  showed exactly 0 at night. This is a deliberate change (accuracy over
  matching the old convention). v4 stores **no flag column** for these two:
  it would be a constant `QC_RECOMPUTED` (8) on every row (decision D1, `QC_SCHEMA.md` §3b); instead
  the files say it in `DATA_DICTIONARY.md`, the figshare README and the Parquet metadata, so nobody mistakes
  the angles for measurements. The input of the cleaning (the `_original` file) has no Azimuth/Elevation at
  all. In the 1 min / 5 min / 60 min files they are the mean of the 1 s values (circular for the azimuth).

## QC flag columns
Flags are merged into the released data files as one `int8` `<column>_qc`
column per treated source column (e.g. `Pressure[mbar]_qc`), not as a
separate companion file, so they survive a plain `pd.read_hdf()` /
`pd.read_parquet()` with nothing else to keep in sync. The code table is in
`QC_SCHEMA.md`; how flags aggregate to coarser resolutions is in
`METHODOLOGY.md`.

## What was wrong with the originally published (v3) files

| Issue in v3 | Finding | Handled in v4 |
|---|---|---|
| `Pressure[mbar]` mostly pegged to round placeholder values | Placeholders are 1000.0/2000.0 blocks plus isolated 3000.0 glitches, and also non-round flatline plateaus (e.g. 997.5 mbar held for an hour). Only 2019-01-16 has genuinely varying pressure. | NaN, flag 2 (§4) |
| `HUMIDITY[%]` above 100 % | A stuck-sensor plateau lasting the whole of 2018-11-17. | NaN, flag 9 (§6) |
| `WIND_DIR[deg]` above 360° | Raw values reach 712° (extra sensor revolutions, almost all 2018-11-17). The 103 out-of-range rows in the v3 *hourly* file cannot be reproduced from the raw file by any averaging — each exceeds its hour's own raw maximum — so that file's wind direction was most likely built from a different or earlier raw version. | wrapped, flag 1 (§3) |
| `Azimuth[deg]` / `Elevation[deg]` ~99.9 % zero | Sparse placeholders, populated on one day only (2019-01-16), computed assuming local time. | dropped, recomputed (§8); no flag column in v4 (D1) |
| Rows not in chronological order | The raw 1-second file stores 457 daily blocks in shuffled order. The grid itself is complete. | sorted (§1): the v4 `_original` file is already sorted |
| `P_Gaia[kW]` zero almost everywhere | Real telemetry exists on two days only; cause of the zeros unresolved. | values kept, flags 7/10 (§7) |

## Open questions and limitations
- **Pressure sentinel completeness.** Both thresholds (exact multiple of 1000;
  flatline ≥ 300 s) are judgment calls. A real but brief excursion to a round
  number, or a short non-round plateau, could still slip through, and a genuine
  reading of exactly 1000.0 mbar on 2019-01-16 would be removed.
  `diagnostics/audit_pressure_sentinels.py` probes both directions.
- **Why `P_Gaia` is zero.** Only DTU SYSLAB's operations or logging records
  could settle "turbine off" versus "channel not logged".
- **Physical bounds are provisional** (§6): validated against what the profiler
  showed, not against site records.
- **Other v3 hourly columns.** Whether `TEMPERATURE`, `HUMIDITY`,
  `WIND_SPEED` and `Pressure` in the v3 hourly file were built from an adjusted
  raw version rather than a plain mean is tested by
  `diagnostics/followup2_diagnostics.py`.
- **Azimuth offset in the v3 column** (~4.72°, constant, on its one populated
  day) was never explained and is moot: those values are not delivered.
