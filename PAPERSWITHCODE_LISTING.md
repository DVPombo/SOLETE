# Papers with Code — dataset listing draft (Session 4)

This file drafts the content for a Papers with Code dataset listing. It is
**not** submitted automatically — PwC dataset pages are created/edited through
their web UI (or a structured YAML in their own `sotabench`/dataset-page
repo), which is a manual maintainer action. Copy the fields below into that
form when ready.

---

## Name

SOLETE

## Full / display name

SOLETE: co-located solar PV, wind, and meteorology dataset (Denmark)

## Short description (for the listing summary field)

A 15-month, high-resolution (down to 1 Hz) dataset of co-located solar PV
power, wind turbine power, and meteorological measurements from a hybrid
power system in Denmark. Includes derived hybrid (PV+wind) power, quality
control (QC) flags on select PV/meteorology columns, and a reproducible
train/val/test split with an accompanying forecasting benchmark (persistence,
climatology, AR, gradient boosting, LSTM/CNN/CNN-LSTM baselines).

## Longer description (for the listing body)

SOLETE combines simultaneous solar PV power, wind turbine power, and weather
station measurements (irradiance, temperature, wind speed, humidity, and
more) collected over roughly 15 months at DTU's Risø campus in Denmark, at
resolutions from 1 second up to 60 minutes. It was originally released as
companion data to a Data in Brief article (Pombo, Gehrke & Bindner, 2022) and
has since grown into a small platform for time-series forecasting research —
this repository adds a QC-flag layer over known sensor issues, a derived
`P_hybrid[kW]` column combining PV and wind, a canonical chronological
train/val/test split (`splits/v1.json`) for reproducible benchmarking, and a
baseline forecasting leaderboard (see `BENCHMARKS.md`) spanning persistence,
smart-persistence, climatology, AR, gradient boosting, and LSTM/CNN/CNN-LSTM
models for both the PV and wind targets.

**Known data characteristic worth flagging for anyone benchmarking on this
dataset:** the wind turbine's power output (`P_Gaia[kW]`) is exactly zero for
over 99.5% of rows in the full record (see `KNOWN_ISSUES.md` #10 and
`splits/README.md`) — most plausibly the turbine was largely out of service
during the recording period (unconfirmed). Any wind-power benchmark result on
this dataset, including the ones in this repo's own leaderboard, should be
read with that near-all-zero target in mind rather than as a measure of
genuine forecasting skill.

## Task tags

- Time Series Forecasting
- Energy Forecasting (renewables / solar power forecasting / wind power
  forecasting)
- Regression
- Short-term power forecasting

## Modality

Tabular time series (multivariate)

## Paper

Pombo, D. V., Gehrke, O., & Bindner, H. W. (2022). SOLETE, a 15-month long
holistic dataset including: Meteorology, co-located wind and solar PV power
from Denmark with various resolutions. *Data in Brief*, 42, 108046.
https://doi.org/10.1016/j.dib.2022.108046

## Data source / download

- Canonical dataset (DTU Data, DOI): https://doi.org/10.11583/DTU.17040767
- Figshare mirror (useful where the DOI resolver is unreachable):
  https://figshare.com/articles/dataset/The_SOLETE_dataset/17040767

## Code repository

https://github.com/DVPombo/SOLETE

## Benchmark summary (from `BENCHMARKS.md`)

Reproducible baseline results on the canonical `splits/v1.json` test block
(2,953 rows, 2019-05-01 → 2019-09-01), scored with `metrics.py` (MAE, RMSE,
nRMSE under both capacity and mean normalization), under two QC settings
(`qc_included`/`qc_excluded`) for the PV target:

- **PV (`P_Solar[kW]`)**: persistence, smart persistence, climatology, AR(p=48),
  gradient boosting (LightGBM, same-timestamp weather features — flagged as
  not directly comparable to the lagged-only models), and LSTM/CNN/CNN-LSTM
  (the neural rows are explicitly labeled reduced-epoch/indicative, not final,
  per the sandbox wall-clock constraint documented in `BENCHMARKS.md`).
- **Wind (`P_Gaia[kW]`)**: the same model family, with the near-all-zero-target
  caveat above attached directly to the table.
- **Hybrid (`P_hybrid[kW]`)**: joint-vs-independent forecasting comparison and
  ramp-rate analysis (see `BENCHMARKS.md`'s hybrid section) — scoped honestly
  as infrastructure/methodology given the sparse wind record, not a positive
  wind-solar complementarity claim.

Full tables, caveats, and reproduction details live in `BENCHMARKS.md` and
`splits/README.md`.

## License note

**Flagged, not resolved** (same open item as Session 3's Hugging Face card —
see `CONTRIBUTING.md`): the repository's top-level `LICENSE` and
`CITATION.cff` say MIT, while several script headers (`Functions.py`,
`MLForecasting.py`) and `requirements.txt` say CC-BY 4.0. When filling in
Papers with Code's license field, don't pick one to resolve the discrepancy —
either point to the maintainer's eventual decision, or mirror this repo's
own approach of noting the inconsistency rather than asserting a single
license. If a future session resolves this, update this listing draft too.

## Homepage

https://github.com/DVPombo/SOLETE
