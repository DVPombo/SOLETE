# SOLETE — Known Data & Code Issues

This file is a single point of reference for data-quality issues (open and fixed) and
one process/housekeeping gap. It does not duplicate `CHANGELOG.md`'s full detail — it
points there (and to git history) for that — but it's a reference specifically for
someone deciding "can I trust this column / this file as-is."

Each entry: what it is, where it lives, current status, what to do in the meantime.

---

## Open data-quality issues 

### 1. `Pressure[mbar]` mostly pegged to round placeholder values in the 60min file — **FIXED**
- **What it is:** In ``SOLETE_Pombo_60min.h5`, 95.5% hold exactly
  `1000.000000` mbar, 4.3% hold exactly `2000.0`, and 2 rows hold `3000.0`.
  Only a few rows carry a plausible, non-round atmospheric pressure value (992–998 mbar).
  `examples/SOLETE_short.h5`'s pressure column, by contrast, varies smoothly and realistically
  (1013.26–1017.47 mbar) across all 24 rows.
- **Where it lives:** `SOLETE_Pombo_60min.h5`, column `Pressure[mbar]`.
- **Current status:** Fixed — The error was not present on the second resolution.
  
---

### 2. `P_Gaia[kW]` near-total zero-degeneracy — root cause unconfirmed (Phase 6) — **Resolved**
- **What it is:** `P_Gaia[kW]` is exactly `0.0` for 99.56% across
  the full real 15-month record. The only non-zero readings are rows on 2018-08-31 and
  on 2019-05-25 (0.44%) — despite thousands of rows elsewhere with wind
  speed above the turbine's 3.5 m/s cut-in. 
- **Two candidate explanations, both unconfirmed, neither preferred by this repo's code
  or data alone:**
  1. **Turbine out of service.** The original Phase 5 write-up's leading guess — physically
     plausible (11 kW research turbines at a single site can sit idle for maintenance,
     grid-connection, or project reasons for long stretches), but nothing in this repo
     (logbook, maintenance record, status column) confirms it either way.
  2. **Resolution/aggregation pipeline artifact.** `docs/RESOLUTIONS.md` already found a related,
     *confirmed-symptom* problem in the same aggregation pipeline: `WIND_DIR[deg]` in this
     same 60min file holds 103 rows (0.94%) above the physically-valid 360° compass range,
     a pattern consistent with a non-circular-mean bug in whatever built the coarser
     resolution files from the finer ones (see `docs/RESOLUTIONS.md` and finding #6 above). No
     aggregation code or finer-resolution (`1sec`/`1min`/`5min`) source file exists in this
     repo to check directly against, for either `WIND_DIR[deg]` or `P_Gaia[kW]` — so this
     explanation is equally unconfirmed, not equally unlikely. A pipeline step that zeroes
     out (rather than mis-averages) a channel under some condition is a different failure
     mode than the angle-wrapping bug, but the same root uncertainty applies: **this repo
     cannot rule in or rule out either explanation from what's available in it.**
- **Current status:** Open. Filed with the same "confirmed symptom / unconfirmed
  mechanism" framing as finding #6, deliberately.
- **What a user should do in the meantime:** Treat any wind or hybrid (`P_hybrid[kW]`)
  result on this dataset as **infrastructure/methodology**, not a demonstrated finding
  about wind behavior or wind-solar complementarity, until this is resolved. 

## Summary table

| # | Issue | File(s) | Status |
|---|---|---|---|
| 1 | `Pressure[mbar]` pegged to round placeholders | `SOLETE_Pombo_60min.h5` | Fixed |
| 2 | `P_Gaia[kW]` near-total zero-degeneracy, root cause unconfirmed | `SOLETE_Pombo_60min.h5` | Resolved |