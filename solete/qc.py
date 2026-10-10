# -*- coding: utf-8 -*-
"""
solete/qc.py -- the platform's thin quality-flag layer.

Since version 4 there is ONE flag vocabulary (`solete/qc_codes.py`) and the dataset
pipeline owns every flag that describes the raw sensor stream. What is left here:

  1. `present_qc_columns` / `check_qc_vocabulary`: read the `<column>_qc` columns a v4 file
     already carries and verify they use the shared vocabulary. Nothing is recomputed or
     overwritten (this is what makes `data_version='v4'` safe to load).
  2. `add_substitution_flag`: the single platform-owned flag, code 6 (QC_MODEL_SUBSTITUTED),
     written on `P_Solar[kW]_qc` by `solete/expansion.py`.
  3. `apply_qc_flags` + `legacy_v3_raw_value_rules`: ONLY for the original v3 files, which were
     never cleaned by the pipeline and have no `_qc` columns. The old platform detection rules are
     kept (same rows flagged as before) but they now emit code 11 (QC_UNTREATED_IMPLAUSIBLE,
     "implausible value kept as recorded"), because the old platform numbers 1 and 3 mean something
     different in the shared vocabulary. A v4 file never goes through them.

Original author: Daniel Vázquez Pombo
email: daniel.vazquez.pombo@gmail.com

Licensed under the MIT License -- see LICENSE at the repo root. If you use this work, please give credit (see CITATION.cff).
"""

import numpy as np

from .qc_codes import (  # noqa: F401  (re-exported: the vocabulary lives in qc_codes)
    QC_OK,
    QC_MODEL_SUBSTITUTED,
    QC_UNTREATED_IMPLAUSIBLE,
    QC_LABELS,
    QC_SEVERITY_ORDER,
    SEVERITY_RANK,
    severity_rank,
)

QC_SUFFIX = "_qc"

# Known placeholder/sentinel values for Pressure[mbar] (KNOWN_ISSUES.md finding #1), v3 files only.
KNOWN_PRESSURE_SENTINELS = {1000.0, 2000.0, 3000.0}
# General safety net: Earth-surface sea-level-pressure record extremes (mbar).
PRESSURE_PLAUSIBLE_RANGE = (870.0, 1085.0)


# ---------------------------------------------------------------------------
# 1. Reading what a v4 file already carries
# ---------------------------------------------------------------------------

def present_qc_columns(data):
    """Names of the `<column>_qc` flag columns that are present in `data` (not the
    resampled `_qc_worst` / `_qc_frac_flagged` ones, and not `_qc_source`)."""
    return [c for c in data.columns if c.endswith(QC_SUFFIX)]


def check_qc_vocabulary(data, columns=None):
    """Return {column: sorted unknown codes} for every present `<column>_qc` column that holds a
    code outside the shared vocabulary (empty dict = all fine). Read-only."""
    known = set(QC_LABELS)
    bad = {}
    for c in (present_qc_columns(data) if columns is None else columns):
        codes = set(np.unique(data[c].to_numpy()).tolist())
        unknown = sorted(codes - known)
        if unknown:
            bad[c] = unknown
    return bad


# ---------------------------------------------------------------------------
# 2. The platform-owned flag (code 6)
# ---------------------------------------------------------------------------

def add_substitution_flag(existing_qc, substituted):
    """Combine an existing `P_Solar[kW]_qc` array (or None) with the substitution boolean.

    Code 6 is platform-owned and recomputed on every expansion, so a 6 already present in
    `existing_qc` is treated as QC_OK first (this is what makes expansion idempotent). Then 6
    is written where `substituted` is True unless the cell already carries a more severe code
    (QC_SEVERITY_ORDER). Returns an int8 array."""
    substituted = np.asarray(substituted, dtype=bool)
    if existing_qc is None:
        base = np.zeros(substituted.shape, dtype=np.int8)
    else:
        base = np.asarray(existing_qc).astype(np.int8, copy=True)
        base[base == QC_MODEL_SUBSTITUTED] = QC_OK
    take = substituted & (severity_rank(base) > SEVERITY_RANK[QC_MODEL_SUBSTITUTED])
    base[take] = QC_MODEL_SUBSTITUTED
    return base


# ---------------------------------------------------------------------------
# 3. v3 only: flags for files the pipeline never cleaned
# ---------------------------------------------------------------------------

def apply_qc_flags(data, rules):
    """
    Apply a list of QC detection rules to `data`, adding one <column>_qc column per distinct
    qc_column named in `rules` (int8, QC_OK where no rule fires).

    rules: list of dicts with
        'column'    source column the detector reads
        'qc_column' optional override, default f"{column}_qc"
        'flag'      code (see solete/qc_codes.py)
        'detector'  callable(pd.Series) -> boolean Series, True where the flag applies
    Rules are processed in severity order (QC_SEVERITY_ORDER, worst first) and a cell that is no
    longer QC_OK is never overwritten, so each cell keeps exactly one code.

    Returns (data, counts) where counts = {qc_column: rows flagged by these rules}.
    """
    ordered_rules = sorted(rules, key=lambda r: SEVERITY_RANK.get(r['flag'], len(SEVERITY_RANK)))

    counts = {}
    for rule in ordered_rules:
        col = rule['column']
        if col not in data.columns:
            print(f"apply_qc_flags: column '{col}' not found, skipping "
                  f"rule for '{rule.get('qc_column', col + '_qc')}'")
            continue

        qc_col = rule.get('qc_column', f"{col}_qc")
        mask = rule['detector'](data[col])

        if qc_col not in data.columns:
            data[qc_col] = np.zeros(len(data), dtype=np.int8)

        apply_mask = mask & (data[qc_col] == QC_OK)
        data.loc[apply_mask, qc_col] = rule['flag']
        counts[qc_col] = counts.get(qc_col, 0) + int(apply_mask.sum())

    return data, counts


def legacy_v3_raw_value_rules(data):
    """
    Raw-value checks for the ORIGINAL v3 files (KNOWN_ISSUES.md findings #1-#4), which carry no
    flags of their own: implausible Pressure/Humidity/WindDir values and the Azimuth/Elevation
    "== 0.0" missingness proxy. The detectors are exactly the ones the platform always used; the
    flag they write is QC_UNTREATED_IMPLAUSIBLE (11): the value is kept as recorded, it is not
    cleaned. Never call this on a v4 file: the pipeline's own flags are authoritative there.
    Only returns rules whose source column exists in `data`.
    """
    flag = QC_UNTREATED_IMPLAUSIBLE
    # Azimuth/Elevation first: keeps the column order the platform always produced.
    candidates = [
        {
            'column': 'Azimuth[deg]',
            'flag': flag,
            'detector': lambda s: s == 0.0,
        },
        {
            'column': 'Elevation[deg]',
            'flag': flag,
            'detector': lambda s: s == 0.0,
        },
        {
            'column': 'Pressure[mbar]',
            'flag': flag,
            'detector': lambda s: s.isin(KNOWN_PRESSURE_SENTINELS)
                | (s < PRESSURE_PLAUSIBLE_RANGE[0])
                | (s > PRESSURE_PLAUSIBLE_RANGE[1]),
        },
        {
            'column': 'HUMIDITY[%]',
            'flag': flag,
            'detector': lambda s: (s > 1.0) | (s < 0.0),
        },
        {
            'column': 'WIND_DIR[deg]',
            'flag': flag,
            'detector': lambda s: (s >= 360.0) | (s < 0.0),
        },
    ]
    return [r for r in candidates if r['column'] in data.columns]
