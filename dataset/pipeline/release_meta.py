# -*- coding: utf-8 -*-
"""
release_meta.py -- what the v4 release files say about themselves: the per-column provenance, the site
constants and the conventions. One definition, used by the Parquet metadata (export_parquet.py), the
manifest and the verification (build_release.py), so they cannot drift apart. The documents
(DATA_DICTIONARY.md, data/figshare_README.txt) describe the same classes in prose; if a class below changes,
change them.

Provenance classes (the words used in the metadata):
  measured                    a value recorded by the instruments, unchanged (the `_original` file)
  measured, cleaned           a measured value after the pipeline's cleaning rules (1 s v4 file)
  resampled from 1 s          a measured/pipeline column aggregated from the cleaned 1 s data
  recomputed by the pipeline  Azimuth/Elevation: pvlib from timestamp and site, replacing the faulty originals
  pipeline flag               a `<column>_qc` code assigned by the cleaning rules (1 s v4 file)
  computed at this resolution a model column (solete.expansion.expand_physical) from this file's own inputs
"""
import datetime as _dt
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from solete.qc_codes import MODEL_DERIVED_COLUMNS, QC_LABELS, SOURCE_LABELS  # noqa: E402
from solar_position import SITE_ALTITUDE_M, SITE_LATITUDE, SITE_LONGITUDE  # noqa: E402

RELEASE_VERSION = "v4"
MEASURED_COLUMNS = ("TEMPERATURE[degC]", "HUMIDITY[%]", "WIND_SPEED[m1s]", "WIND_DIR[deg]", "GHI[kW1m2]",
                    "POA Irr[kW1m2]", "P_Gaia[kW]", "P_Solar[kW]", "Pressure[mbar]")
ANGLE_COLUMNS = ("Azimuth[deg]", "Elevation[deg]")
FILE_RE = re.compile(r"^SOLETE_Pombo_(1sec|1min|5min|60min)(_original)?_v4$")

INTERVAL_LABEL = ("start of interval: a bucket labelled T covers [T, T+period) for the 1 min, 5 min and 60 min files; "
                  "the 1 s file has one instant per row")
ANGLE_CONVENTION = ("Azimuth[deg]: 0 = south, east negative, west positive, range [-180, 180); Elevation[deg]: apparent "
                    "elevation above the horizon, negative at night (not clipped). pvlib NREL SPA, UTC timestamps, "
                    "site constants below. In the 1 min / 5 min / 60 min files they are the mean of the 1 s values "
                    "(circular mean for azimuth).")


def parse_stem(stem):
    """('1sec'|'1min'|'5min'|'60min', is_original) for a release file stem, or (None, False)."""
    m = FILE_RE.match(stem)
    return (m.group(1), bool(m.group(2))) if m else (None, False)


def column_provenance(columns, resolution=None, original=False):
    """{column: provenance string} for the columns of one release file."""
    coarse = resolution in ("1min", "5min", "60min")
    out = {}
    for c in columns:
        if c == "timestamp":
            out[c] = "time index, UTC; " + ("start of the interval" if coarse else "instant")
        elif c in MEASURED_COLUMNS:
            if original:
                out[c] = "measured (raw v3 value, unchanged)"
            elif coarse:
                out[c] = "resampled from 1 s (mean" + ("; circular mean" if c == "WIND_DIR[deg]" else "") + ") after cleaning"
            else:
                out[c] = ("measured, cleaned (values changed only where the column's _qc flag is non-zero; "
                          "P_Solar[kW] and P_Gaia[kW] are never altered)")
        elif c in ANGLE_COLUMNS:
            out[c] = ("resampled from 1 s (mean of the recomputed values)" if coarse
                      else "recomputed by the pipeline (pvlib from timestamp and site)")
        elif c in MODEL_DERIVED_COLUMNS or c == "P_Solar_clean[kW]":
            out[c] = "computed at this resolution (solete.expansion.expand_physical from this file's own inputs)"
        elif c.endswith("_qc_worst"):
            out[c] = "resampled from 1 s pipeline flags (highest-severity code in the bucket; NaN = empty bucket)"
        elif c.endswith("_qc_frac_flagged"):
            out[c] = "resampled from 1 s pipeline flags (fraction of seconds not QC_OK; NaN = empty bucket)"
        elif c.endswith("_qc"):
            out[c] = "pipeline flag (assigned per second by the cleaning rules)"
        else:
            out[c] = "unknown"
    return out


def file_metadata(source_file, n_rows, columns, keep_naive=False, extra=None):
    """The dict stored under b'solete' in the Parquet file metadata (and listed in the manifest)."""
    import pvlib
    stem = Path(source_file).stem
    res, original = parse_stem(stem)
    meta = {
        "dataset": "SOLETE",
        "version": RELEASE_VERSION,
        "file_role": ("original (raw 1 s, sorted, nine measured columns, nothing cleaned)" if original
                      else "cleaned and expanded" if res else "unrecognised file name"),
        "resolution": res,
        "source_file": Path(source_file).name,
        "n_rows": int(n_rows),
        "timestamp_timezone": "naive (values are UTC)" if keep_naive else "UTC",
        "timestamp_unit": "ns",
        "timestamp_label": INTERVAL_LABEL,
        "site": {"latitude": SITE_LATITUDE, "longitude": SITE_LONGITUDE, "altitude_m": SITE_ALTITUDE_M,
                 "name": "DTU Risoe campus SYSLAB, Denmark"},
        "angle_convention": ANGLE_CONVENTION,
        "pvlib_version": pvlib.__version__,
        "build_date_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d"),
        "qc_columns": [c for c in columns if c.endswith("_qc") or c.endswith("_qc_worst")],
        "qc_codes": {str(k): v for k, v in QC_LABELS.items()},
        "qc_code_8_note": "reserved, not emitted by any v4 column",
        "qc_source_codes": {str(k): v for k, v in SOURCE_LABELS.items()},
        "column_provenance": column_provenance(["timestamp"] + list(columns), res, original),
        "documentation": "dataset/docs/DATA_DICTIONARY.md, dataset/docs/QC_SCHEMA.md, data/figshare_README.txt",
    }
    if extra:
        meta.update(extra)
    return meta


def encode(meta):
    return {b"solete": json.dumps(meta, ensure_ascii=True).encode("utf-8")}
