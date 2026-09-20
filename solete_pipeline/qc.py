# -*- coding: utf-8 -*-
"""
Part of the solete_pipeline package -- split out of the original
Functions.py (Phase 7, Session 8) for independent testability.
See Functions.py (kept as a re-export shim) and CONTRIBUTING.md for
why this split happened and how the modules relate to each other.

Original author: Daniel Vázquez Pombo
email: daniel.vazquez.pombo@gmail.com

Licensed under the MIT License -- see LICENSE at the repo root. If you use this work, please give credit (see CITATION.cff).
"""

import pandas as pd

# ---------------------------------------------------------------------------
# QC flag layer. See QC_SCHEMA.md at the repo root for the full
# design rationale (flag semantics, mutually-exclusive vs bitmask decision,
# precedence order, and why each detection rule looks the way it does).
# ---------------------------------------------------------------------------

QC_VALID = 0
QC_MISSING = 1
QC_SENSOR_ERROR = 2                    # reserved, no detector wired up yet
QC_PHYSICALLY_IMPLAUSIBLE = 3
QC_INTERPOLATED = 4                    # reserved, no detector wired up yet
QC_AGGREGATION_AFFECTED_BY_GAPS = 5    # reserved, no detector wired up yet
QC_SUSPECTED_CURTAILMENT_OR_MODEL_SUBSTITUTED = 6

# Precedence order (highest first) used by apply_qc_flags to resolve the rare
# case where more than one rule targets the same <column>_qc cell. None of the
# rules below actually collide today (see QC_SCHEMA.md section 2), this is
# just so a future rule doesn't have to invent a tie-break from scratch.
QC_FLAG_PRECEDENCE = [
    QC_MISSING,
    QC_SENSOR_ERROR,
    QC_PHYSICALLY_IMPLAUSIBLE,
    QC_AGGREGATION_AFFECTED_BY_GAPS,
    QC_INTERPOLATED,
    QC_SUSPECTED_CURTAILMENT_OR_MODEL_SUBSTITUTED,
]

# Known placeholder/sentinel values for Pressure[mbar] (KNOWN_ISSUES.md
# finding #1). Kept as an explicit, easily-extended set rather than baked into
# the detector logic -- see QC_SCHEMA.md section 5 for why a range check alone
# can't catch the 1000.0 case.
KNOWN_PRESSURE_SENTINELS = {1000.0, 2000.0, 3000.0}
# General safety net: Earth-surface sea-level-pressure record extremes (mbar).
# Catches a future out-of-scale sentinel even if it's not in the set above.
PRESSURE_PLAUSIBLE_RANGE = (870.0, 1085.0)


def apply_qc_flags(data, rules):
    """
    Apply a list of QC detection rules to `data`, adding one <column>_qc
    column per distinct qc_column named in `rules`. General-purpose: Task 2.2
    uses it for the three physically-implausible-value checks, Task 2.3 reuses
    it unchanged for the azimuth/elevation missingness check and the
    P_Solar_model_substituted mapping.

    Parameters
    ----------
    data : DataFrame
        Mutated in place -- one <column>_qc column is added/updated per rule.
    rules : list of dict
        Each dict:
            'column'     : source column name the detector reads from.
            'qc_column'  : optional override for the flag column name,
                           defaults to f"{column}_qc".
            'flag'       : int QC flag value to assign where the detector
                           returns True (see the QC_* constants above).
            'detector'   : callable(pd.Series) -> boolean Series, True where
                           that flag applies.
        Process rules in QC_FLAG_PRECEDENCE order (highest priority first) if
        more than one rule ever targets the same qc_column, so a
        lower-priority rule never overwrites a cell a higher-priority rule
        already flagged. Cells no rule touches default to QC_VALID.

    Returns
    -------
    data : DataFrame
        Same object passed in (mutated), returned for convenience/chaining.
    counts : dict
        {qc_column: number_of_rows_flagged_by_this_rule} -- for sanity checks
        and logging. If two rules share a qc_column, later (lower-precedence)
        rules only count cells they actually got to flag (i.e. that weren't
        already claimed by a higher-precedence rule), matching what ends up
        on disk.
    """
    ordered_rules = sorted(
        rules,
        key=lambda r: QC_FLAG_PRECEDENCE.index(r['flag'])
        if r['flag'] in QC_FLAG_PRECEDENCE else len(QC_FLAG_PRECEDENCE),
    )

    counts = {}
    for rule in ordered_rules:
        col = rule['column']
        if col not in data.columns:
            print(f"apply_qc_flags: column '{col}' not found, skipping "
                  f"rule for '{rule.get('qc_column', col + '_qc')}'")
            continue

        qc_col = rule.get('qc_column', f"{col}_qc")
        flag = rule['flag']
        mask = rule['detector'](data[col])

        if qc_col not in data.columns:
            data[qc_col] = QC_VALID

        # Mutually exclusive: don't clobber a cell a higher-precedence rule
        # (processed earlier, since we sorted by precedence) already flagged.
        available = data[qc_col] == QC_VALID
        apply_mask = mask & available
        data.loc[apply_mask, qc_col] = flag

        counts[qc_col] = counts.get(qc_col, 0) + int(apply_mask.sum())

    return data, counts


def build_raw_value_qc_rules(data):
    """
    QC rules for findings #1-#4 (KNOWN_ISSUES.md / QC_SCHEMA.md section 3):
    Pressure/Humidity/WindDir physically-implausible values, and the
    Azimuth/Elevation missingness proxy. Only returns rules whose source
    column actually exists in `data` -- SOLETE_short.h5 has no
    Azimuth[deg]/Elevation[deg], for example, and apply_qc_flags would skip
    them anyway, but building the list this way keeps the caller's log clean.
    """
    candidates = [
        {
            'column': 'Pressure[mbar]',
            'flag': QC_PHYSICALLY_IMPLAUSIBLE,
            'detector': lambda s: s.isin(KNOWN_PRESSURE_SENTINELS)
                | (s < PRESSURE_PLAUSIBLE_RANGE[0])
                | (s > PRESSURE_PLAUSIBLE_RANGE[1]),
        },
        {
            'column': 'HUMIDITY[%]',
            'flag': QC_PHYSICALLY_IMPLAUSIBLE,
            'detector': lambda s: (s > 1.0) | (s < 0.0),
        },
        {
            'column': 'WIND_DIR[deg]',
            'flag': QC_PHYSICALLY_IMPLAUSIBLE,
            'detector': lambda s: (s >= 360.0) | (s < 0.0),
        },
        {
            'column': 'Azimuth[deg]',
            'flag': QC_MISSING,
            'detector': lambda s: s == 0.0,
        },
        {
            'column': 'Elevation[deg]',
            'flag': QC_MISSING,
            'detector': lambda s: s == 0.0,
        },
    ]
    return [r for r in candidates if r['column'] in data.columns]


def build_substitution_qc_rule(source_col='P_Solar_model_substituted',
                                qc_col='P_Solar[kW]_qc'):
    """
    QC rule for finding #6: folds the existing P_Solar_model_substituted
    boolean (added in Phase 0.5, computed in ExpandSOLETE from
    Pac >= 1.5*P_Solar[kW]) into the unified <column>_qc convention, without
    touching the substitution logic itself. True -> flag 6
    (suspected_curtailment_or_model_substituted), False -> flag 0 (valid).
    """
    return {
        'column': source_col,
        'qc_column': qc_col,
        'flag': QC_SUSPECTED_CURTAILMENT_OR_MODEL_SUBSTITUTED,
        'detector': lambda s: s == True,  
    }


