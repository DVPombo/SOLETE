# Papers with Code dataset listing — draft content

This file is **drafted content only**, for the maintainer to copy from when
manually creating/editing the actual listing on paperswithcode.com (via its
web UI, or a structured submission to its own `sotabench`/dataset-page repo)
— nothing here is hosted or submitted automatically from this repository.

---

## Name

SOLETE

## Full / display title

SOLETE: co-located wind and solar PV power dataset from Denmark

## Short description (for the listing summary field)

A 15-month, multi-resolution time-series dataset of co-located meteorological,
wind-power, and solar-PV-power measurements from SYSLAB, a distributed-energy-
resources test laboratory at DTU Wind and Energy Systems, Denmark. Combines an
11 kW Gaia wind turbine and a 10 kW-class PV array/inverter with a
meteorological station on the same site, recorded at up to 1 Hz and also
provided at 1-minute, 5-minute, and hourly resolutions. Originally introduced
as complementary material to a Data in Brief article, the accompanying code
repository has since grown into a small forecasting-benchmark platform (QC
flagging, a canonical chronological train/val/test split, and baseline-through-
neural-network forecasting results — see "Benchmark" below).

## Longer description (for the listing's full description field)

The SOLETE dataset pairs simultaneous weather, wind-power, and solar-PV-power
measurements from a single, real, distributed energy-resources site, rather
than combining wind and solar from separate locations after the fact — the
"co-located" part of its name. It spans roughly 15 months (2018-06-01 through
2019-09-01) and is distributed at several resolutions (1-second, 1-minute,
5-minute, and 60-minute); the 60-minute file is the one this repository's own
benchmark and QC work is run against, and is the only resolution shipped in
the repo itself for size reasons (see "Data access" below for the rest).

Columns include ambient temperature, relative humidity, wind speed and
direction, global horizontal and plane-of-array irradiance, barometric
pressure, measured wind-turbine output (`P_Gaia[kW]`), and measured PV output
(`P_Solar[kW]`) — see `DATA_DICTIONARY.md` in the code repository for the full,
verified-against-the-real-files column-by-column reference (units, ranges,
and known data-quality caveats per column), rather than duplicating that
description here.

**Known data-quality caveats worth surfacing in the listing itself** (full
detail in `KNOWN_ISSUES.md` and `DATA_DICTIONARY.md`):
- `P_Gaia[kW]` (wind) is exactly zero for 99.56% of the 60-minute file's rows
  — the turbine appears to have been out of service for nearly the entire
  record (unconfirmed root cause). Any wind-power benchmark result on this
  dataset is overwhelmingly a score on an all-zero target and should be read
  accordingly.
- A minority of `P_Solar[kW]` rows are a model-substituted value rather than
  the raw sensor reading (flagged via the `P_Solar_model_substituted` /
  `P_Solar[kW]_qc` columns added in this repository's QC layer — not present
  in the original DTU Data release).
- `Pressure[mbar]` contains known non-physical sentinel values
  (1000/2000/3000 mbar) on a large share of rows, also flagged via QC columns
  in this repository rather than cleaned in the underlying data itself.

## Task category tags

- Time-Series Forecasting
- Energy Forecasting
(Papers with Code's own tag vocabulary may phrase these slightly differently
at submission time — check current tag names in the web UI rather than
assuming these exact strings still exist.)

## Modality

Time series (tabular, timestamped, multivariate).

## Links

- **Paper:** Pombo, D. V., Gehrke, O., & Bindner, H. W. (2022). SOLETE, a
  15-month long holistic dataset including: Meteorology, co-located wind and
  solar PV power from Denmark with various resolutions. *Data in Brief*, 42,
  108046. https://doi.org/10.1016/j.dib.2022.108046
- **Code repository:** https://github.com/DVPombo/SOLETE
- **Canonical data record (DTU Data / figshare):**
  https://doi.org/10.11583/DTU.17040767.v3 (direct figshare mirror, useful if
  the DOI redirect is blocked for you:
  https://figshare.com/articles/dataset/The_SOLETE_dataset/17040767)

## Data access

Papers with Code dataset pages typically link out to where the actual files
live rather than hosting them — link to the DTU Data / figshare record above
as the download source, **not** to this GitHub repository (which ships only
the small `SOLETE_short.h5` sample and the `SOLETE_Pombo_60min.h5` file used
for this repo's own benchmark; the full multi-resolution dataset, including
1-second and 1-minute resolutions, is on DTU Data / figshare only).

## Benchmark description

This repository (`BENCHMARKS.md`) maintains a leaderboard-style benchmark on
top of the 60-minute-resolution file, using a fixed chronological
train/val/test split (`splits/v1.json`; test block 2019-05-01 through
2019-09-01, 2,953 rows) so results across models are comparable. One-step-
ahead (1-hour horizon) point forecasts are scored with MAE, RMSE, and nRMSE
(under both a fixed-capacity and a mean-of-target normalization —
`metrics.py`), for both the PV (`P_Solar[kW]`) and wind (`P_Gaia[kW]`)
targets, each under two QC settings (all rows as-is, versus excluding rows
flagged as PV-model-substituted).

Models currently benchmarked: naive persistence, "smart" (24h-lag)
persistence, hour-of-day climatology, an autoregressive AR(p) model, gradient
boosting (LightGBM, using same-timestamp weather features — flagged in
`BENCHMARKS.md` as not a fair comparison against the lagged-input-only rows),
and LSTM / CNN / CNN-LSTM neural networks (the latter three explicitly flagged
in `BENCHMARKS.md` as reduced-epoch / indicative results, not final, due to
sandbox runtime limits when they were produced).

**Caveat that must travel with any Papers with Code benchmark entry sourced
from this repo:** every wind-target (`P_Gaia[kW]`) result is a score against
an almost-entirely-zero target (see above) — near-perfect wind metrics
reflect that degeneracy, not forecasting skill, and Papers with Code listings
that surface only a single best-score number per task should not present a
wind result from this dataset without that caveat attached.

## Licensing note for the listing

There is a known, unresolved licensing inconsistency in the code repository
(documented in its `CONTRIBUTING.md`): the top-level `LICENSE` file and
`CITATION.cff` both say MIT, but several script headers (e.g. `Functions.py`,
`MLForecasting.py`) and `requirements.txt` say CC-BY 4.0 instead. This
concerns the *code*, not the dataset's own DTU Data licensing terms (see the
DTU Data / figshare record for the data's own license). If Papers with Code's
submission form requires picking a single license, flag this inconsistency to
the maintainer rather than guessing one — same convention followed for the
Hugging Face dataset card (`huggingface/README.md`) in this repo.

## Citation

```
@article{Pombo2022SOLETE,
  title={SOLETE, a 15-month long holistic dataset including: Meteorology, co-located wind and solar PV power from Denmark with various resolutions},
  author={Pombo, Daniel V{\'a}zquez and Gehrke, Oliver and Bindner, Henrik W.},
  journal={Data in Brief},
  volume={42},
  pages={108046},
  year={2022},
  publisher={Elsevier}
}
```
