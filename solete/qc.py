"""Platform helpers for consuming v4 QC columns and adding code 6."""

import numpy as np
import pandas as pd

from .qc_codes import QC_MODEL_SUBSTITUTED, QC_OK, QC_SEVERITY_ORDER


def qc_columns(data):
    """Return QC columns already present in a SOLETE frame."""
    return [
        column for column in data.columns
        if column.endswith("_qc") or column.endswith("_qc_worst")
    ]


def add_substitution_flag(data, substituted, qc_col="P_Solar[kW]_qc"):
    """Merge the per-resolution model-substitution flag without losing QC."""
    substituted = pd.Series(substituted, index=data.index, dtype=bool)
    if qc_col in data.columns:
        existing = data[qc_col].copy()
        existing = existing.mask(existing == QC_MODEL_SUBSTITUTED, QC_OK)
    else:
        existing = pd.Series(QC_OK, index=data.index, dtype=np.int8)

    rank = {code: position for position, code in enumerate(QC_SEVERITY_ORDER)}
    existing_rank = existing.map(rank).fillna(len(rank)).to_numpy()
    substitution_rank = rank[QC_MODEL_SUBSTITUTED]
    use_substitution = substituted.to_numpy() & (substitution_rank < existing_rank)
    data[qc_col] = np.where(use_substitution, QC_MODEL_SUBSTITUTED, existing).astype(np.int8)
    return data[qc_col]