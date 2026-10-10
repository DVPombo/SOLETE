# -*- coding: utf-8 -*-
"""
solete/expansion.py -- the deterministic "physical" expansion of a SOLETE table.

`expand_physical(df, pv_info)` adds every derived column that is a pure function of ONE row's
own cleaned inputs: no rolling window, no neighbouring row, no horizon, no assumption about the
time step. That is what lets the same function run on a 1-second month, a 5-minute year or the
hourly file, on any slice of rows (chunking gives bit-identical results), and again on a file that
already contains its model columns (idempotent).

Columns added (see docs: DATA_DICTIONARY.md, METHODOLOGY.md "Model columns are per resolution"):

    Pac, Pdc, TempModule, TempCell     King's PV performance model (kW, kW, degC, degC)
    P_Solar[kW]_qc                     existing flag column, plus code 6 (QC_MODEL_SUBSTITUTED) where
                                       Pac >= 1.5 * P_Solar[kW]  AND  Pac > 0   (NaN input -> not flagged).
                                       The second condition is on the stored Pac (values <= 0.001 are 0), so a
                                       night row (model 0, measurement 0) is NOT a substitution.
                                       (There is no separate boolean column: it was identical to `== 6`.)
    P_Solar_clean[kW]                  P_Solar[kW] with the model value where substituted,
                                       then values <= 0.001 set to 0.  P_Solar[kW] itself is
                                       NEVER touched: the measurement stays as measured.
    P_hybrid[kW]                       P_Solar_clean[kW] + P_Gaia[kW]
    P_hybrid[kW]_qc, _qc_source        flag inherited from the worse of P_Solar[kW]_qc and
                                       P_Gaia[kW]_qc (or P_Gaia[kW]_qc_worst in resampled files) by the shared severity order
                                       (solete/qc_codes.py); ties go to P_Solar. `_qc_source` is an int8 code saying which
                                       one it came from (qc_codes.SOURCE_LABELS: 0 none, 1 P_Solar, 2 P_Gaia).

`Pac` is stored with values <= 0.001 set to 0 (as ExpandSOLETE always did); the ">= 1.5 * P_Solar" test
uses the unrounded value, "Pac > 0" the stored one. Before the 0-vs-0 night rows were flagged too
(`0 >= 1.5 * 0`); P_Solar_clean[kW] and P_hybrid[kW] are bit-identical either way, only the flag differs.

Not here, on purpose: `TempModule_RP` (the Rincon-Pombo model carries the module temperature of the
previous row, so it is not row-wise) and everything that depends on `Control_Var` or a horizon
(HoursOfDay, MeanPrevH, StdPrevH, MeanWindSpeedPrevH, StdWindSpeedPrevH) stay in
`solete.preprocessing.ExpandSOLETE`, which calls `expand_physical` first.

Resolution rule: these columns are computed at each resolution from that resolution's own cleaned
inputs. They are never averaged up from a finer resolution (see qc_codes.MODEL_DERIVED_COLUMNS).

Memory: temporaries are ~25 float64 arrays per chunk row, so the default chunk of 1,000,000 rows
needs roughly 200 MB on top of the output columns. For a frame that does not fit in RAM (the full
1-second file) call it on slices of about a month and write each slice out; do not hold the result
of the whole file.
"""

import numpy as np
import pandas as pd

from .physics import pv_model_arrays
from .qc_codes import QC_OK, SOURCE_NONE, SOURCE_SOLAR, SOURCE_WIND, severity_rank
from .qc import add_substitution_flag

SUBSTITUTION_FACTOR = 1.5      # P_Solar is replaced by the model where Pac >= 1.5 * P_Solar
ZERO_SMOOTHING = 0.001         # kW; values <= this are set to exactly 0

COL_IRR = 'POA Irr[kW1m2]'
COL_TEMP = 'TEMPERATURE[degC]'
COL_WS = 'WIND_SPEED[m1s]'
COL_SOLAR = 'P_Solar[kW]'
COL_WIND = 'P_Gaia[kW]'
REQUIRED_INPUTS = (COL_IRR, COL_TEMP, COL_WS, COL_SOLAR, COL_WIND)

SOLAR_QC = 'P_Solar[kW]_qc'
WIND_QC = 'P_Gaia[kW]_qc'
WIND_QC_WORST = 'P_Gaia[kW]_qc_worst'   # what the resampled (1min/5min/1h) pipeline files carry instead
HYBRID_QC = 'P_hybrid[kW]_qc'
HYBRID_SOURCE = 'P_hybrid[kW]_qc_source'

# column order produced (and, for new columns, appended to the frame)
PHYSICAL_COLUMNS = ('Pac', 'Pdc', 'TempModule', 'TempCell',
                    SOLAR_QC, 'P_Solar_clean[kW]', 'P_hybrid[kW]', HYBRID_QC, HYBRID_SOURCE)

DEFAULT_CHUNK_ROWS = 1_000_000


def _expand_arrays(df, pv_info):
    """Core, row-wise computation on one block of rows. Returns a dict of NumPy arrays
    (the source tag as an int8, see qc_codes.SOURCE_LABELS)."""
    missing = [c for c in REQUIRED_INPUTS if c not in df.columns]
    if missing:
        raise KeyError(f"expand_physical needs columns {missing}")
    solar = df[COL_SOLAR].to_numpy(dtype=np.float64)
    wind = df[COL_WIND].to_numpy(dtype=np.float64)

    pac_raw, pdc, tmod, tcell = pv_model_arrays(
        df[COL_IRR].to_numpy(), df[COL_TEMP].to_numpy(), df[COL_WS].to_numpy(), pv_info)

    with np.errstate(invalid='ignore'):
        pac = np.where(pac_raw <= ZERO_SMOOTHING, 0, pac_raw)
        # the model is clearly above the measurement AND the model is producing (excludes 0 vs 0 at night); NaN -> False
        substituted = (pac_raw >= SUBSTITUTION_FACTOR * solar) & (pac > 0)
        clean = np.where(substituted, pac_raw, solar)
        clean = np.where(clean <= ZERO_SMOOTHING, 0, clean)             # NaN stays NaN
    hybrid = clean + wind

    solar_qc = add_substitution_flag(df[SOLAR_QC].to_numpy() if SOLAR_QC in df.columns else None, substituted)
    if WIND_QC in df.columns:
        wind_qc = df[WIND_QC].to_numpy().astype(np.int8)
    elif WIND_QC_WORST in df.columns:        # resampled files: the worst code in the bucket (NaN = empty bucket -> OK)
        wind_qc = df[WIND_QC_WORST].fillna(QC_OK).to_numpy().astype(np.int8)
    else:
        wind_qc = np.zeros(len(df), dtype=np.int8)
    r_solar, r_wind = severity_rank(solar_qc), severity_rank(wind_qc)
    solar_first = r_solar <= r_wind                                      # ties -> P_Solar
    hybrid_qc = np.where(solar_first, solar_qc, wind_qc).astype(np.int8)
    source = np.where(hybrid_qc == QC_OK, SOURCE_NONE, np.where(solar_first, SOURCE_SOLAR, SOURCE_WIND)).astype(np.int8)

    return {
        'Pac': pac, 'Pdc': pdc, 'TempModule': tmod, 'TempCell': tcell,
        SOLAR_QC: solar_qc, 'P_Solar_clean[kW]': clean,
        'P_hybrid[kW]': hybrid, HYBRID_QC: hybrid_qc, HYBRID_SOURCE: source,
    }


def compute_physical(df, pv_info, chunk_rows=DEFAULT_CHUNK_ROWS):
    """Return a NEW DataFrame (same index as `df`) holding only the PHYSICAL_COLUMNS; `df` is not
    modified. Result is identical whatever `chunk_rows` is (every rule is row-wise).
"""
    n = len(df)
    chunk_rows = n if (chunk_rows is None or chunk_rows <= 0) else int(chunk_rows)
    out = None
    for start in range(0, max(n, 1), max(chunk_rows, 1)):
        block = df.iloc[start:start + chunk_rows]
        arrays = _expand_arrays(block, pv_info)
        if out is None:
            out = {k: np.empty(n, dtype=v.dtype) for k, v in arrays.items()}
        for k, v in arrays.items():
            out[k][start:start + len(block)] = v
        del arrays
    if out is None:   # empty frame
        out = {k: np.empty(0, dtype=t) for k, t in zip(
            PHYSICAL_COLUMNS, [np.float64] * 4 + [np.int8, np.float64, np.float64, np.int8, np.int8])}
    return pd.DataFrame({k: pd.Series(out[k], index=df.index) for k in PHYSICAL_COLUMNS}, index=df.index)


def expand_physical(df, pv_info, chunk_rows=DEFAULT_CHUNK_ROWS):
    """Add the PHYSICAL_COLUMNS to `df` in place (and return it). Existing columns of the same
    name (a file that was expanded before) are recomputed and replaced: running this on its own
    output reproduces it exactly. `P_Solar[kW]` and every other input column is left untouched.
    A code 6 already present in `P_Solar[kW]_qc` is platform-owned and recomputed, not trusted."""
    new = compute_physical(df, pv_info, chunk_rows=chunk_rows)
    for col in new.columns:
        # positional assignment: indexes may hold duplicate or unsorted timestamps (the v3 files do)
        df[col] = new[col].array
    return df
