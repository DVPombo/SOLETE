import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from _data import short_sample_path
from dataset.pipeline.resample_solete import resample_dataframe
from solete.expansion import (
    PHYSICAL_COLUMNS,
    expand_physical,
    iter_expanded_hdf_slices,
    iter_hdf_slices,
)
from solete.io import import_PV_WT_data, import_SOLETE_data, import_SOLETE_sample
from solete.physics import PV_Performance_Model
from solete.qc import add_substitution_flag, qc_columns
from solete.qc_codes import (
    QC_GLITCH_LONG_UNTREATED_NAN,
    QC_LABELS,
    QC_MODEL_SUBSTITUTED,
    QC_OK,
    QC_SEVERITY_ORDER,
)


def _control(**overrides):
    control = {
        "resolution": "1sec",
        "SOLETE_builvsimport": "Build",
        "SOLETE_save": False,
        "OriginalFeatures": [],
        "PossibleFeatures": [],
    }
    control.update(overrides)
    return control


def _sample():
    return pd.read_hdf(short_sample_path())


def test_shared_qc_vocabulary_includes_model_substitution():
    assert QC_MODEL_SUBSTITUTED == 6
    assert QC_LABELS[QC_MODEL_SUBSTITUTED] == "model_substituted"
    assert QC_SEVERITY_ORDER.index(QC_GLITCH_LONG_UNTREATED_NAN) < QC_SEVERITY_ORDER.index(6)


def test_qc_columns_reads_direct_and_resampled_flags():
    frame = pd.DataFrame(columns=["Pressure[mbar]_qc", "P_Gaia[kW]_qc_worst", "value"])
    assert qc_columns(frame) == ["Pressure[mbar]_qc", "P_Gaia[kW]_qc_worst"]


def test_substitution_flag_preserves_more_severe_existing_flag():
    frame = pd.DataFrame({"P_Solar[kW]_qc": [QC_GLITCH_LONG_UNTREATED_NAN, QC_OK]})
    add_substitution_flag(frame, [True, True])
    assert frame["P_Solar[kW]_qc"].tolist() == [QC_GLITCH_LONG_UNTREATED_NAN, 6]


def test_substitution_flag_clears_stale_code_six():
    frame = pd.DataFrame({"P_Solar[kW]_qc": [QC_MODEL_SUBSTITUTED]})
    add_substitution_flag(frame, [False])
    assert frame["P_Solar[kW]_qc"].iloc[0] == QC_OK


def test_expand_physical_preserves_measured_solar_and_is_idempotent(tmp_path):
    pv_info, _ = import_PV_WT_data()
    frame = _sample()
    frame["Pressure[mbar]_qc"] = np.int8(2)
    frame["P_Gaia[kW]_qc"] = np.int8(7)
    measured = frame["P_Solar[kW]"].copy()

    expand_physical(frame, pv_info)
    first = frame[list(PHYSICAL_COLUMNS)].copy()
    path = tmp_path / "expanded_v4_like.h5"
    frame.to_hdf(path, key="DATA")
    reloaded = pd.read_hdf(path, key="DATA")
    expand_physical(reloaded, pv_info)

    pd.testing.assert_series_equal(reloaded["P_Solar[kW]"], measured)
    pd.testing.assert_frame_equal(reloaded[list(PHYSICAL_COLUMNS)], first)
    assert (reloaded["Pressure[mbar]_qc"] == 2).all()


def test_expand_physical_slice_output_matches_full_run():
    pv_info, _ = import_PV_WT_data()
    source = _sample()
    full = expand_physical(source.copy(), pv_info)
    chunked = pd.concat([
        expand_physical(source.iloc[start:start + 5].copy(), pv_info)
        for start in range(0, len(source), 5)
    ])
    pd.testing.assert_frame_equal(chunked, full)


def test_fixed_hdf_slices_match_pandas_and_expand_independently():
    pv_info, _ = import_PV_WT_data()
    path = short_sample_path()
    expected = pd.read_hdf(path)

    sliced = pd.concat(iter_hdf_slices(path, chunk_rows=5))
    expanded = pd.concat(iter_expanded_hdf_slices(path, pv_info, chunk_rows=5))

    pd.testing.assert_frame_equal(sliced, expected)
    pd.testing.assert_frame_equal(expanded, expand_physical(expected.copy(), pv_info))


def test_table_hdf_slices_match_pandas(tmp_path):
    expected = _sample()
    path = tmp_path / "table.h5"
    expected.to_hdf(path, key="DATA", format="table")

    actual = pd.concat(iter_hdf_slices(path, chunk_rows=5))

    pd.testing.assert_frame_equal(actual, expected)


def test_hybrid_qc_uses_canonical_severity_order():
    pv_info, _ = import_PV_WT_data()
    frame = _sample().iloc[:2].copy()
    frame["P_Gaia[kW]_qc"] = [QC_GLITCH_LONG_UNTREATED_NAN, QC_OK]

    expand_physical(frame, pv_info)

    assert frame["P_hybrid[kW]_qc"].iloc[0] == QC_GLITCH_LONG_UNTREATED_NAN
    assert frame["P_hybrid[kW]_qc_source"].iloc[0] == "P_Gaia[kW]"


def test_expandsolete_matches_legacy_sample_physical_values():
    pv_info, wt_info = import_PV_WT_data()
    source = _sample()
    pac, pdc, temp_module, temp_cell = PV_Performance_Model(source, pv_info)
    substituted = pac >= 1.5 * source["P_Solar[kW]"]
    legacy_solar = np.where(substituted, pac, source["P_Solar[kW]"])
    legacy_solar = np.where(legacy_solar <= 0.001, 0.0, legacy_solar)

    actual = import_SOLETE_sample(short_sample_path(), _control(), pv_info, wt_info)

    np.testing.assert_array_equal(actual["Pac"], np.where(pac <= 0.001, 0.0, pac))
    pd.testing.assert_series_equal(actual["Pdc"], pdc, check_names=False)
    pd.testing.assert_series_equal(actual["TempModule"], temp_module, check_names=False)
    pd.testing.assert_series_equal(actual["TempCell"], temp_cell, check_names=False)
    np.testing.assert_array_equal(actual["P_Solar[kW]"], legacy_solar)
    np.testing.assert_array_equal(actual["P_hybrid[kW]"], legacy_solar + source["P_Gaia[kW]"])
    np.testing.assert_array_equal(
        actual["P_Solar[kW]_qc"], np.where(substituted, QC_MODEL_SUBSTITUTED, QC_OK)
    )


def test_import_v4_preserves_pipeline_flags(monkeypatch, tmp_path):
    pv_info, wt_info = import_PV_WT_data()
    source = _sample()
    source["Pressure[mbar]_qc"] = np.int8(2)
    path = tmp_path / "SOLETE_Pombo_1sec_cleaned_v4.h5"
    source.to_hdf(path, key="DATA")
    monkeypatch.setattr("solete.io.find_data_file", lambda *args, **kwargs: path)

    actual = import_SOLETE_data(_control(data_version="v4"), pv_info, wt_info)

    assert (actual["Pressure[mbar]_qc"] == 2).all()
    pd.testing.assert_series_equal(actual["P_Solar[kW]"], actual["P_Solar_clean[kW]"], check_names=False)


def test_resampling_drops_model_columns_and_never_carries_code_six():
    index = pd.date_range("2020-01-01", periods=120, freq="s")
    frame = pd.DataFrame({
        "P_Solar[kW]": np.linspace(0.0, 1.0, len(index)),
        "P_Gaia[kW]": 0.0,
        "Pac": 1.0,
        "P_Solar_clean[kW]": 1.0,
        "P_Solar[kW]_qc": QC_MODEL_SUBSTITUTED,
        "P_Gaia[kW]_qc": QC_MODEL_SUBSTITUTED,
    }, index=index)

    result = resample_dataframe(frame, "1min")

    assert "Pac" not in result
    assert "P_Solar_clean[kW]" not in result
    assert "P_Solar[kW]_qc_worst" not in result
    assert (result["P_Gaia[kW]_qc_worst"] == QC_OK).all()
    assert (result["P_Gaia[kW]_qc_frac_flagged"] == 0.0).all()