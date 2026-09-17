# QC_SCHEMA.md — SOLETE quality-control flag layer

Phase 2 design doc. Turns the findings in `KNOWN_ISSUES.md` into an explicit, queryable
per-column flag layer. This is a **design + mapping** document; implementation lives in
`Functions.py` (`apply_qc_flags`), `tests/test_qc_flags.py`, and
`scripts/availability_report.py`.

## 1. Flag value set

```
0 = valid
1 = missing
2 = sensor_error
3 = physically_implausible
4 = interpolated
5 = aggregation_affected_by_gaps
6 = suspected_curtailment_or_model_substituted
```

`sensor_error` (2), `interpolated` (4), and `aggregation_affected_by_gaps` (5) are reserved
by this schema but have no detection rule wired up yet — none of the six Phase 1 findings
needs them today. They're kept in the enum so a future check (e.g. a real hardware fault
code, or a resample that spans a data gap) has a home without renumbering everything else.

## 2. Bitmask vs. mutually exclusive — **decision: mutually exclusive**

**Bitmask** would let a single cell carry more than one flag at once (e.g. `missing` *and*
`aggregation_affected_by_gaps` simultaneously). Pros: more honest when problems genuinely
co-occur on the same cell; extensible without redefining combinations. Cons: raw values
stop being human-readable without a decode helper (`5` could mean two different things
depending on which bits it is), `value_counts()` on the raw column no longer gives clean
per-category counts, and it's more code to build and to consume correctly.

**Mutually exclusive** (chosen): each `<column>_qc` cell holds exactly one flag value.
Checked against the real data: every one of the six findings maps to a **different
column**, each with a single-purpose rule — there's no case in this dataset where one cell
of one column needs two flags at once (and no NaNs exist in the affected raw columns that
would create a missing-vs-implausible collision). A plain integer column is trivial to read,
matches the codebase's existing convention (`P_Solar_model_substituted` is a plain
boolean, not bit-packed), and a documented precedence order (below) resolves any future tie
deterministically without needing bitmask machinery we don't currently have a use for.

**Precedence order**, applied if a future rule set ever produces more than one candidate
flag for the same cell (highest priority first, first match wins):
`missing` > `sensor_error` > `physically_implausible` > `aggregation_affected_by_gaps` >
`interpolated` > `suspected_curtailment_or_model_substituted` > `valid`.
Rationale: "we don't have the value at all" and "the sensor told us it's broken" are
stronger claims than "the value looks physically off," which is stronger than a downstream
processing artifact. None of the current rules actually exercise this order today — it's
here so the next rule added doesn't have to invent a tie-break from scratch.

## 3. Finding → column → flag → detection rule

| # | `KNOWN_ISSUES.md` finding | Column(s) | Flag | Detection rule |
|---|---|---|---|---|
| 1 | Pressure sentinels (95.5% @ 1000.0, 4.3% @ 2000.0, 2 rows @ 3000.0) | `Pressure[mbar]` | 3 `physically_implausible` | `value in KNOWN_PRESSURE_SENTINELS` (currently `{1000.0, 2000.0, 3000.0}`) **OR** `value < 870` **OR** `value > 1085` (Earth-surface record extremes, as a general safety net — see §5) |
| 2 | Humidity >1.0 (1.7%, up to 2.7) | `HUMIDITY[%]` | 3 `physically_implausible` | `value > 1.0 or value < 0.0` |
| 3 | Wind dir ≥360° (0.94%, up to 639.34°) | `WIND_DIR[deg]` | 3 `physically_implausible` | `value >= 360.0 or value < 0.0` |
| 4 | Azimuth/Elevation ~99.9% zero, real values one day only | `Azimuth[deg]`, `Elevation[deg]` | 1 `missing` | `value == 0.0` (see §6 — these are a later, user-added derivation, not part of the original DTU release) |
| 5 | Row order not chronological on disk (60min file) | *file-level, no `_qc` column* | — | Not represented in this schema. It's not a per-cell value problem — every value is correct once sorted — so it doesn't fit the `<column>_qc` pattern. **Decision (maintainer, 2026-09-10): documented here as a known caveat only; no code change this phase.** Revisit after the current GitHub pass — likely candidates are a `.sort_index()` in `import_SOLETE_data()`, or fixing wherever the 60min file is built from the source resolution. |
| 6 | `P_Solar_model_substituted` (Phase 0.5, ~38%/~21%) | `P_Solar[kW]` | 6 `suspected_curtailment_or_model_substituted` | existing boolean `Pac >= 1.5 * P_Solar[kW]` (computed in `ExpandSOLETE`, unchanged) mapped `True → 6`, `False → 0` |

### Note on finding #1's flag label

Calling the `1000.0` case `physically_implausible` is a slight simplification worth being
explicit about: 1000 mbar is an entirely ordinary atmospheric pressure on its own — it's
suspicious only because **95.5% of all rows hit that exact literal value**, which no real
sensor with real noise does. `2000.0` and `3000.0`, by contrast, are genuinely
off-scale for Earth's surface regardless of frequency. Both get bucketed under flag `3`
for consistency with findings #2/#3 (operationally, all three mean "don't trust this
value"), but the *detection rule* for `1000.0` specifically is an exact-literal match, not a
range check — the range check alone would never catch it.

## 4. Naming convention

`<column>_qc`, e.g. `Pressure[mbar]_qc`, `HUMIDITY[%]_qc`, `WIND_DIR[deg]_qc`,
`Azimuth[deg]_qc`, `Elevation[deg]_qc`, `P_Solar[kW]_qc`. Checked against
`inspect_dataset.py`'s column listing for both real files — no collisions with any existing
or derived column name.

## 5. Sentinel generality (Task 2.2 open question)

Hardcoding `{1000.0, 2000.0, 3000.0}` is brittle if a fourth sentinel shows up in a
different file later — but a purely range-based rule can't catch `1000.0` at all, since
it's a plausible value in isolation (see §3 note above). Resolution: keep the known-literal
set as the primary, documented detection path (it's *why* we know these three rows are bad,
and it's the one Phase 1 actually verified), and add a broad range check
(`< 870 or > 1085` mbar, global sea-level pressure record extremes) as a secondary,
general-purpose safety net that would catch a future out-of-scale sentinel without needing
a code change. `KNOWN_PRESSURE_SENTINELS` is a module-level constant in `Functions.py`. so
extending it later is a one-line change, not a rewrite.

## 6. Azimuth/Elevation — provenance note

Per the maintainer: these two columns were **not** part of the original DTU release
described in the SOLETE paper (which lists only temperature, humidity, pressure, wind
speed/direction, GHI, POA irradiance, and WT/PV power) — they were added later using a
separate Python library from GPS coordinates and timestamp. An independent check against
`pvlib`'s solar-position calculation for the one populated day (2019-01-16, site
55.6867°N 12.0985°E) shows elevation tracking within ~0.1–1° at midday, and azimuth
(south-referenced convention) tracking with a consistent ~11–12° offset for the 8
well-above-horizon rows — i.e. the populated day looks like a genuine calculation, not
noise, modulo a convention/time-basis difference worth revisiting separately. The QC
concern here is coverage (99.9% zero / not computed), not correctness of the populated day
— hence flag `1` (`missing`), not `3` (`physically_implausible`). **Revisit alongside the
row-order issue after the current GitHub pass** — out of scope for this phase to
recompute/backfill these columns.

## 7. Behavior under expand/resample/export

`ExpandSOLETE`'s `list_expansion` mechanism (which currently tracks `Pac`, `Pdc`,
`TempModule`, `TempCell`, `P_Solar_model_substituted`, etc., only for a print statement)
does **not** by itself protect a column from being dropped on a save→Import round trip.
The actual drop logic in `import_SOLETE_data()`'s `'Import'` branch checks column names
against `Control_Var['PossibleFeatures']` — a user-maintained literal list in
`MLForecasting.py` — and drops anything not on it. **This is already happening today**:
`P_Solar_model_substituted` is absent from the example `PossibleFeatures` list, so a
Build→save→Import round trip with the shipped config silently drops it. New `_qc` columns
would inherit the exact same exposure.

Given that gap can't be closed from inside `Functions.py` alone (it requires the *user's own*
`Control_Var` literal to list the new columns), the design compensates by placement, per the
decision above:
- The raw-value checks (findings #1–#4) are computed inside `import_SOLETE_data()` itself,
  directly from columns that exist in the raw file, in **both** the `Build` and `Import`
  branches. This makes them self-healing: even if the persisted `_qc` columns get dropped on
  Import (because they weren't listed in `PossibleFeatures`), they're regenerated on the
  next load from the still-present raw columns.
- The substitution-flag mapping (finding #6) is computed inside `ExpandSOLETE()`, immediately
  after `Pac` exists — it's the only place the underlying boolean can be derived from.
- `CHANGELOG.md` and this document call out explicitly that anyone using `'Import'` mode
  with a custom `Control_Var['PossibleFeatures']` needs to add the new `_qc` column names
  (and `P_Solar_model_substituted`, which was already exposed to this risk before this
  phase) to that list, or they'll be dropped on load from a saved expanded file — the same
  underlying gap, not something this phase introduces.

## 8. Derived/combined columns — QC inheritance (Task 6.1)

Phase 6 adds the first *derived-from-two-columns* target, `P_hybrid[kW]` (=
`P_Solar[kW] + P_Gaia[kW]`, see `DATA_DICTIONARY.md`). Nothing above covers how a
combined column's QC flag should be built from its constituents' flags — this section
closes that gap.

**Checked first: does `P_Gaia[kW]` have a QC column of its own?** No. None of the six
Phase 1/2 findings target it, `build_raw_value_qc_rules()` has no rule keyed on it, and
`ExpandSOLETE()` never touches it. This isn't an oversight being fixed here — the raw
`P_Gaia[kW]` values that exist all pass the kind of physically-implausible/missing checks
finding #1–#4 look for; the column's real problem (near-total zero-degeneracy, see
`KNOWN_ISSUES.md` #10) isn't a per-row data-quality defect, it's a documented dataset-level
caveat that lives in `splits/README.md` and `KNOWN_ISSUES.md`, not a `_qc` flag. Flagging
every zero row as suspect would be wrong — most of them may be genuinely zero output on a
calm day; the issue is the *pattern*, not any individual value.

**Rule (implemented in `ExpandSOLETE`, `Functions.py`):** `P_hybrid[kW]_qc` takes
whichever constituent's flag ranks higher in `QC_FLAG_PRECEDENCE` (the same precedence
order `apply_qc_flags` already uses for same-cell collisions); `0` (`valid`) if both
constituents are valid. A companion column, `P_hybrid[kW]_qc_source`, records which
constituent produced the flag (`'P_Solar[kW]'`, `'P_Gaia[kW]'`, or `'none'`), so a QC-flag
value alone never leaves a reader guessing which half of the sum triggered it.

Because `P_Gaia[kW]` has no QC rule today, `P_hybrid[kW]_qc_source` is currently always
either `'P_Solar[kW]'` or `'none'` in practice — real data confirms this: on the full
60min record, all 4,204 `P_hybrid[kW]_qc == 6` rows trace back to `P_Solar[kW]_qc`, zero to
wind. The combination logic itself doesn't hardcode that asymmetry, though — it's written
against whichever `'<constituent>_qc'` columns exist on `data`, so if a wind QC rule is
ever added (e.g. under `QC_AGGREGATION_AFFECTED_BY_GAPS`, reserved flag `5`, once/if the
resolution-aggregation question in `KNOWN_ISSUES.md` #10 is resolved), `P_hybrid[kW]_qc`
starts reflecting it immediately, no code change required here.
