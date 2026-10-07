"""Deterministic physical expansion and bounded-memory HDF iteration."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .physics import PV_Performance_Model
from .qc import add_substitution_flag
from .qc_codes import QC_OK, QC_SEVERITY_ORDER


PHYSICAL_COLUMNS = (
    "Pac",
    "Pdc",
    "TempModule",
    "TempCell",
    "P_Solar_model_substituted",
    "P_Solar_clean[kW]",
    "P_Solar[kW]_qc",
    "P_hybrid[kW]",
    "P_hybrid[kW]_qc",
    "P_hybrid[kW]_qc_source",
)


def _decode_hdf_labels(values):
    return [value.decode("utf-8") if isinstance(value, bytes) else value for value in values]


def iter_hdf_slices(path, key="DATA", chunk_rows=2_678_400):
    """Yield bounded-memory row slices from pandas fixed or table HDF files."""
    path = Path(path)
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    with pd.HDFStore(path, mode="r") as store:
        normalized_key = key if str(key).startswith("/") else f"/{key}"
        storer = store.get_storer(normalized_key)
        format_type = storer.format_type
        nrows = storer.nrows

    if format_type == "table":
        for start in range(0, nrows, chunk_rows):
            yield pd.read_hdf(path, key=key, start=start, stop=min(start + chunk_rows, nrows))
        return

    group_name = str(key).lstrip("/")
    with h5py.File(path, mode="r") as h5:
        group = h5[group_name]
        columns = _decode_hdf_labels(group["axis0"][:])
        nrows = len(group["axis1"])
        blocks = []
        block_number = 0
        while f"block{block_number}_values" in group:
            block_columns = _decode_hdf_labels(group[f"block{block_number}_items"][:])
            blocks.append((block_columns, group[f"block{block_number}_values"]))
            block_number += 1

        for start in range(0, nrows, chunk_rows):
            stop = min(start + chunk_rows, nrows)
            data = {}
            for block_columns, values in blocks:
                block = values[start:stop]
                if block.shape[0] != stop - start:
                    block = block.T
                for position, column in enumerate(block_columns):
                    data[column] = block[:, position]
            index = pd.to_datetime(group["axis1"][start:stop], unit="ns")
            yield pd.DataFrame(data, index=index)[columns]


def iter_expanded_hdf_slices(path, pv_info, key="DATA", chunk_rows=2_678_400):
    """Yield independently expanded HDF slices without modifying the input."""
    for frame in iter_hdf_slices(path, key=key, chunk_rows=chunk_rows):
        yield expand_physical(frame, pv_info)


def _constituent_qc(data, column):
    for suffix in ("_qc", "_qc_worst"):
        qc_column = f"{column}{suffix}"
        if qc_column in data.columns:
            return data[qc_column].to_numpy()
    return np.full(len(data), QC_OK, dtype=np.int8)


def expand_physical(data, pv_info):
    """Add deterministic row-wise columns without changing measured values."""
    pac, pdc, temp_module, temp_cell = PV_Performance_Model(data, pv_info)
    data["Pac"] = np.where(pac.to_numpy() <= 0.001, 0.0, pac.to_numpy())
    data["Pdc"] = pdc
    data["TempModule"] = temp_module
    data["TempCell"] = temp_cell

    measured_solar = data["P_Solar[kW]"].to_numpy()
    substituted = pac.to_numpy() >= 1.5 * measured_solar
    data["P_Solar_model_substituted"] = substituted
    clean_solar = np.where(substituted, pac.to_numpy(), measured_solar)
    data["P_Solar_clean[kW]"] = np.where(clean_solar <= 0.001, 0.0, clean_solar)
    add_substitution_flag(data, substituted)

    data["P_hybrid[kW]"] = data["P_Solar_clean[kW]"] + data["P_Gaia[kW]"]
    solar_qc = _constituent_qc(data, "P_Solar[kW]")
    wind_qc = _constituent_qc(data, "P_Gaia[kW]")
    severity_rank = {code: position for position, code in enumerate(QC_SEVERITY_ORDER)}
    unknown_rank = len(severity_rank)
    solar_rank = pd.Series(solar_qc).map(severity_rank).fillna(unknown_rank).to_numpy()
    wind_rank = pd.Series(wind_qc).map(severity_rank).fillna(unknown_rank).to_numpy()
    solar_wins = (solar_qc != QC_OK) & (solar_rank <= wind_rank)
    wind_wins = (wind_qc != QC_OK) & ~solar_wins
    data["P_hybrid[kW]_qc"] = np.where(
        solar_wins, solar_qc, np.where(wind_wins, wind_qc, QC_OK)
    ).astype(np.int8)
    data["P_hybrid[kW]_qc_source"] = np.where(
        solar_wins, "P_Solar[kW]", np.where(wind_wins, "P_Gaia[kW]", "none")
    )
    return data