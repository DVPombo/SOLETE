---
pretty_name: "SOLETE: Co-located Wind and Solar PV Power from Denmark"
license: other
license_name: "see-licensing-information-section"
language:
  - en
tags:
  - timeseries
  - energy
  - solar-power
  - wind-power
  - forecasting
  - renewable-energy
task_categories:
  - time-series-forecasting
size_categories:
  - 10K<n<100K
---

# Dataset Card for SOLETE

This is a **discoverability card only**. SOLETE's canonical, citable home is
**DTU Data / figshare** (see below) — this card does not host the dataset files
themselves, only describes them and points to where to actually get them. See
"Where to get the data" for why, and for two working links (one DOI, one direct
figshare link, kept as separate options because the DOI redirect has been reported
as unreachable from behind some regional firewalls).

## Table of Contents

- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Where to get the data](#where-to-get-the-data)
  - [Supported Tasks](#supported-tasks)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Known Data-Quality Issues](#known-data-quality-issues)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)

## Dataset Description

- **Homepage / canonical dataset record:** https://doi.org/10.11583/DTU.17040767.v3
- **Direct figshare link (mirror of the same record, use if the DOI redirect is
  blocked for you):** https://figshare.com/articles/dataset/The_SOLETE_dataset/17040767
- **Code repository (loading/processing/benchmarking code, not the data itself):**
  https://github.com/DVPombo/SOLETE
- **Paper:** Pombo, D. V., Gehrke, O., & Bindner, H. W. (2022). SOLETE, a 15-month
  long holistic dataset including: Meteorology, co-located wind and solar PV power
  from Denmark with various resolutions. *Data in Brief*, 42, 108046.
  https://doi.org/10.1016/j.dib.2022.108046
- **Point of Contact:** Daniel Vázquez Pombo — daniel.vazquez.pombo@gmail.com

### Dataset Summary

SOLETE is a ~15-month (2018-06-01 to 2019-09-01) time series dataset recorded at
SYSLAB, a distributed-energy-resources test laboratory at DTU Wind and Energy
Systems, Denmark. It co-locates meteorological measurements (temperature, humidity,
pressure, wind speed/direction, irradiance) with power output from an 11 kW Gaia
wind turbine and a 10 kW-class PV array/inverter at the same site, at up to four
resolutions (1 second, 1 minute, 5 minute, and 1 hour). It was built to support
research and teaching in solar/wind power forecasting with machine learning,
particularly physics-informed approaches, and has a companion code repository (see
above) with a QC-flag layer, a documented train/val/test split, and a small
reproducible forecasting benchmark built on top of it.

### Where to get the data

**Do not expect to download the actual `.h5` files from this Hugging Face repo.**
This card exists purely so SOLETE is findable where ML/forecasting practitioners
increasingly search first; the dataset's real, versioned, citable home stays DTU
Data (mirrored on figshare). Get the files from one of:

- DOI (may redirect through DTU's own landing page):
  https://doi.org/10.11583/DTU.17040767.v3
- Direct figshare link (same underlying record; use this one if the DOI redirect
  doesn't load for you — this has been reported from behind some regional network
  firewalls, e.g. in China):
  https://figshare.com/articles/dataset/The_SOLETE_dataset/17040767

The code repository's `SOLETE_short.h5` (a 24-row, 1-day sample) and
`SOLETE_Pombo_60min.h5` (the full ~15-month record at 1-hour resolution) are the
two files this card's schema description below was actually verified against; see
[`examples/`](https://github.com/DVPombo/SOLETE/tree/main/examples) in the code
repository for small bundled sample files if you just want to try the dataset's
shape before downloading the full thing from DTU Data/figshare.

### Supported Tasks

- `time-series-forecasting`: point and (with the code repo's Phase 7 addition)
  probabilistic forecasting of solar PV power (`P_Solar[kW]`), wind power
  (`P_Gaia[kW]`), and their sum (`P_hybrid[kW]`), typically from lagged target
  values and/or same-timestamp weather. The code repository's `BENCHMARKS.md`
  documents a reproducible baseline suite (persistence, climatology/AR, gradient
  boosting, LSTM/CNN) against a fixed train/val/test split
  (`splits/v1.json`) — see there for exact numbers before quoting any as a
  state-of-the-art claim; none are presented as SOTA, only as reproducible
  reference points.

### Languages

Not applicable — this is a numeric sensor time series with no natural-language
content; column headers and documentation are in English.

## Dataset Structure

### Data Instances

Each row is one timestamp with co-located weather and power readings. Real first
row of `SOLETE_short.h5` (hourly resolution):

```
Timestamp: 2018-06-01 00:00:00
TEMPERATURE[degC]: 14.165157
HUMIDITY[%]: 0.700000
WIND_SPEED[m1s]: 1.306307
WIND_DIR[deg]: 109.657127
GHI[kW1m2]: 0.000000
POA Irr[kW1m2]: 0.000000
P_Gaia[kW]: 0.037694
P_Solar[kW]: 0.000000
Pressure[mbar]: 1017.473418
```
(exact column set depends on file/resolution — the 60-minute file additionally
has `Azimuth[deg]`/`Elevation[deg]`; see Data Fields below.)

### Data Fields

The full, verified-against-real-files field-by-field description — units, observed
ranges, per-column caveats, and which of the two repo-bundled files each column
appears in — lives in the code repository's
[`DATA_DICTIONARY.md`](https://github.com/DVPombo/SOLETE/blob/main/DATA_DICTIONARY.md)
and is **not duplicated here** to avoid the two documents drifting apart. In
summary, the raw columns are:

`TEMPERATURE[degC]`, `HUMIDITY[%]`, `WIND_SPEED[m1s]`, `WIND_DIR[deg]`,
`GHI[kW1m2]`, `POA Irr[kW1m2]`, `P_Gaia[kW]` (wind power), `P_Solar[kW]` (PV
power), `Pressure[mbar]`, and (60-minute file only) `Azimuth[deg]` /
`Elevation[deg]`.

The code repository's processing step (`ExpandSOLETE()`) adds further derived
columns on top of these — a substitution-tracking flag, per-column QC flags, and
a derived `P_hybrid[kW] = P_Solar[kW] + P_Gaia[kW]` column with its own QC
inheritance — documented in the same `DATA_DICTIONARY.md` and in
[`QC_SCHEMA.md`](https://github.com/DVPombo/SOLETE/blob/main/QC_SCHEMA.md). Those
are pipeline outputs, not part of the raw distributed files.

### Data Splits

The raw dataset as distributed via DTU Data/figshare is not pre-split. The code
repository defines and documents one canonical, reproducible train/validation/test
split for benchmarking purposes
([`splits/v1.json`](https://github.com/DVPombo/SOLETE/blob/main/splits/v1.json),
with the full reproduction recipe and rationale in
[`splits/README.md`](https://github.com/DVPombo/SOLETE/blob/main/splits/README.md)).
That split is a benchmarking convention layered on top of the dataset by the code
repository, not an attribute of the raw dataset release itself.

## Dataset Creation

### Curation Rationale

Built to support (and be transparent about the methodology behind) a series of
papers on physics-informed machine learning for co-located solar/wind power
forecasting — see Citation Information below — and, per the maintainer, to help
newcomers learn time-series forecasting fundamentals with real, imperfect
sensor data rather than a cleaned toy dataset.

### Source Data

Recorded at SYSLAB, DTU Wind and Energy Systems, Denmark, using a meteorological
station, an 11 kW Gaia wind turbine, and a 10 kW-class PV array/inverter,
transferred to a central server (paper-sourced; exact sensor make/model and
calibration records are not published in the code repository or this card).

#### Initial Data Collection and Normalization

Sensors sample at 1 Hz; the released dataset additionally provides 1-minute,
5-minute, and 1-hour aggregates (see Data Splits/Structure above). The exact
per-variable aggregation method (e.g. whether wind direction uses a plain or
circular mean) is not verified in the code repository — see
[`RESOLUTIONS.md`](https://github.com/DVPombo/SOLETE/blob/main/RESOLUTIONS.md)
for what is and isn't confirmed there.

### Annotations

None — this is raw/aggregated sensor data, not a human-annotated dataset. The
code repository's QC flags (`<column>_qc`) are automated rule-based data-quality
markers computed after the fact, not annotations of the original release.

### Personal and Sensitive Information

None identified. This is environmental/energy sensor data from a research test
site; it contains no personal data.

## Considerations for Using the Data

### Known Data-Quality Issues

The code repository's
[`KNOWN_ISSUES.md`](https://github.com/DVPombo/SOLETE/blob/main/KNOWN_ISSUES.md)
is the maintained, single source of truth for this and is **not duplicated here**
since it's actively revised. As of this card's writing it documents, among other
things: `Pressure[mbar]` mostly pegged to round placeholder values in the 60-minute
file, `HUMIDITY[%]` occasionally exceeding its valid [0,1] fractional range,
`WIND_DIR[deg]` occasionally exceeding 360°, and `P_Gaia[kW]` (wind power) being
exactly zero for over 99% of the record with the mechanism unconfirmed. Anyone
using this dataset for wind or hybrid wind+solar forecasting should read that
file's wind-related entries before treating a wind result as a general finding
about the site.

### Other Known Limitations

Only 1-hour-resolution files are bundled in the code repository for real,
verified-against-code documentation; the 1-second/1-minute/5-minute files
described in the source paper are only available from DTU Data/figshare directly
and have not been independently verified against the code repository's
documentation.

## Additional Information

### Dataset Curators

Daniel Vázquez Pombo (Technical University of Denmark at the time of the original
release; see the code repository's `README.md` for the copyright history across
versions), with co-authors Oliver Gehrke and Henrik W. Bindner on the originating
paper.

### Licensing Information

**There is a known, unresolved licensing inconsistency in the code repository**
(documented in its
[`CONTRIBUTING.md`](https://github.com/DVPombo/SOLETE/blob/main/CONTRIBUTING.md)):
the repository's top-level `LICENSE` file and `CITATION.cff` both say MIT, but
several script headers (e.g. `Functions.py`, `MLForecasting.py`) and
`requirements.txt` say CC-BY 4.0 instead. That inconsistency concerns the *code*.
For the *dataset itself*, check the license field on the DTU Data/figshare record
linked above, which is the authoritative source — it is not necessarily the same
license as the code. This card intentionally does not assert a single license
identifier for either the code or the data to avoid guessing past what's
documented; if you need a definitive answer for either, ask the maintainer (see
Point of Contact above) rather than assuming.

### Citation Information

```bibtex
@article{pombo2022solete,
  title={SOLETE, a 15-month long holistic dataset including: Meteorology, co-located wind and solar PV power from Denmark with various resolutions},
  author={Pombo, Daniel Vazquez and Gehrke, Oliver and Bindner, Henrik W},
  journal={Data in Brief},
  volume={42},
  pages={108046},
  year={2022},
  publisher={Elsevier}
}
```

If citing the code/benchmarking repository specifically rather than the dataset
itself, see its `README.md` for the separate software citation
(DOI: 10.11583/DTU.17040626).
