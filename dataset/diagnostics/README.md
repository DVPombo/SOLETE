# Diagnostics

Read-only scripts used to find and confirm the problems documented in
[`../docs/CLEANING_DECISIONS.md`](../docs/CLEANING_DECISIONS.md). You do not need them to
use the dataset or to reproduce the cleaning; they are here so the findings can be checked.
Each prints delimited JSON report blocks (`=====BEGIN SOLETE REPORT: <name>=====`) and
changes nothing on disk. They import helpers from `../pipeline/`, so run them from anywhere.

| Script | Question it answers | Input |
|---|---|---|
| `data_quality_profile.py` | Anything fishy in any column? Per-column stats, zero fractions, longest constant run, most frequent values, wind speed vs. turbine power consistency | a 1 s file |
| `followup_diagnostics.py` | Sections A–E: is the raw 1 s grid complete, are wind-direction values above 360° real, which days is turbine power real, what timezone are the timestamps in, other oddities | raw 1 s file, v3 hourly file, a resampled hourly file |
| `followup2_diagnostics.py` | Sections F–G: were the v3 hourly temperature/humidity/wind speed/pressure built from a differently adjusted raw version? | raw 1 s file, v3 hourly file, optionally the cleaned file |
| `audit_pressure_sentinels.py` | Is the pressure placeholder detector complete, and does it remove genuine readings? | cleaned 1 s file |
| `compare_resolutions.py` | Does a freshly resampled file agree with a published one, once interval labelling is accounted for? | a resampled file, a v3 file |

```
python dataset/diagnostics/data_quality_profile.py SOLETE_Pombo_1sec_v4.h5 --key DATA
```
