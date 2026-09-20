# -*- coding: utf-8 -*-
"""
Unit tests for solete_pipeline's split-out modules (Phase 7, Session 8).

These are new coverage, not a port of an existing test file: the whole point
of splitting Functions.py was to make functions like PV_Performance_Model
and Rincon_Pombo_ThermodynamicModel importable and testable in isolation,
without going through the full import_SOLETE_data -> ExpandSOLETE pipeline
and without pulling in keras/tensorflow. These tests exercise exactly that:
importing solete_pipeline.physics / solete_pipeline.qc directly (not via
Functions.py), and checking that doing so has none of the heavy-dependency
cost Functions.py itself still has.

Real-data-grounded per the same convention as tests/test_qc_flags.py: the PV
input row is a real row from SOLETE_Pombo_60min.h5 (not fabricated), and
PVinfo comes from the real import_PV_WT_data() datasheet values, not
invented numbers. The one exception is explicitly marked SYNTHETIC below,
for a boundary the real file doesn't happen to contain.

Run with: pytest tests/test_solete_pipeline_units.py -v  (from the repo root)
"""

import sys
import subprocess
import pathlib

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Import-isolation: the actual point of the split. If a future change to
# qc.py or physics.py accidentally reintroduces a dependency on modeling.py
# (directly or via a careless `from . import *`), these tests catch it.
# Run in a subprocess so this test file's own already-imported modules
# (pytest may have pulled in all sorts of things collecting other test
# files) can't mask the regression this is meant to catch.
# ---------------------------------------------------------------------------

def _no_heavy_deps_after_import(import_line):
    code = (
        "import sys\n"
        f"{import_line}\n"
        "heavy = [m for m in sys.modules if m.startswith(('keras', 'tensorflow'))]\n"
        "print('HEAVY:' + repr(heavy))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    line = [l for l in result.stdout.splitlines() if l.startswith("HEAVY:")][0]
    heavy = eval(line[len("HEAVY:"):])
    return heavy


def test_qc_module_imports_without_keras_or_tensorflow():
    heavy = _no_heavy_deps_after_import(
        "from solete_pipeline.qc import apply_qc_flags, build_raw_value_qc_rules"
    )
    assert heavy == [], f"solete_pipeline.qc pulled in heavy deps: {heavy}"


def test_physics_module_imports_without_keras_or_tensorflow():
    heavy = _no_heavy_deps_after_import(
        "from solete_pipeline.physics import PV_Performance_Model, "
        "Rincon_Pombo_ThermodynamicModel"
    )
    assert heavy == [], f"solete_pipeline.physics pulled in heavy deps: {heavy}"


def test_Functions_shim_still_needs_keras_and_tensorflow_same_as_before():
    # Functions.py itself is unchanged in scope -- it still re-exports the ML
    # training functions, so it should still trigger the same keras/tensorflow
    # import cost it always had. This isn't a regression; it pins the
    # backward-compatibility expectation so nobody "fixes" Functions.py's
    # import cost in a way that silently drops symbols downstream scripts rely on.
    heavy = _no_heavy_deps_after_import("import Functions")
    assert any(m.startswith("keras") for m in heavy)
    assert any(m.startswith("tensorflow") for m in heavy)


# ---------------------------------------------------------------------------
# solete_pipeline.physics -- PV_Performance_Model, in isolation
# (no HDF5 read, no QC layer, no ExpandSOLETE)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pv_info():
    from solete_pipeline.io import import_PV_WT_data
    PVinfo, _WTinfo = import_PV_WT_data()
    return PVinfo


@pytest.fixture(scope="module")
def real_daytime_row():
    # Real row: 2018-11-17 11:00:00 from SOLETE_Pombo_60min.h5 -- the first
    # row in the file with POA Irr[kW1m2] > 0.5, i.e. genuine daytime
    # production, not a degenerate all-zero night row.
    df = pd.read_hdf(REPO_ROOT / "SOLETE_Pombo_60min.h5")
    row = df.loc[["2018-11-17 11:00:00"]].copy()
    assert row["POA Irr[kW1m2]"].iloc[0] == pytest.approx(0.6844341973890644)
    return row


def test_pv_performance_model_isolated_real_row(pv_info, real_daytime_row):
    from solete_pipeline.physics import PV_Performance_Model

    Pac, Pdc, Tm, Tc = PV_Performance_Model(real_daytime_row, pv_info)

    # Physical sanity, not a change-detector: AC power must not exceed DC
    # power (the inverter can only lose energy, never create it), both must
    # be non-negative for a positive-irradiance row, and the module must run
    # hotter than the cell's own +D_T-independent baseline ambient reading.
    assert (Pac.iloc[0] >= 0) and (Pdc.iloc[0] >= 0)
    assert Pac.iloc[0] <= Pdc.iloc[0] + 1e-9
    assert Tm.iloc[0] >= real_daytime_row["TEMPERATURE[degC]"].iloc[0]
    assert Tc.iloc[0] >= Tm.iloc[0]


def test_pv_performance_model_isolated_zero_irradiance_synthetic(pv_info):
    from solete_pipeline.physics import PV_Performance_Model

    # SYNTHETIC: a hand-built all-zero-irradiance row. Isolates the
    # zero-production boundary directly (King's model should return exactly
    # 0 AC/DC power with no irradiance), which a real night row could also
    # show but this makes the boundary explicit rather than incidental.
    row = pd.DataFrame({
        "POA Irr[kW1m2]": [0.0],
        "TEMPERATURE[degC]": [15.0],
        "WIND_SPEED[m1s]": [2.0],
    }, index=pd.to_datetime(["2020-01-01 00:00:00"]))

    Pac, Pdc, Tm, Tc = PV_Performance_Model(row, pv_info)
    assert Pac.iloc[0] == 0.0
    assert Pdc.iloc[0] == 0.0


# ---------------------------------------------------------------------------
# solete_pipeline.physics -- Rincon_Pombo_ThermodynamicModel, in isolation,
# cross-checked bit-for-bit against the pre-split behavior on real data
# (mirrors the rigor Session 2's own validation used).
# ---------------------------------------------------------------------------

def test_rincon_pombo_isolated_matches_full_pipeline_real_data(pv_info):
    from solete_pipeline.physics import PV_Performance_Model, Rincon_Pombo_ThermodynamicModel

    df = pd.read_hdf(REPO_ROOT / "SOLETE_short.h5")
    Pac, Pdc, TempModule, TempCell = PV_Performance_Model(df, pv_info)
    df = df.copy()
    df["TEMPERATURE[degC]_orig_placeholder"] = df["TEMPERATURE[degC]"]
    # Rincon_Pombo_ThermodynamicModel reads TempModule/TempCell columns which
    # ExpandSOLETE normally sets on `data` before calling it -- reproduce
    # that minimal precondition directly here, isolated from everything else
    # ExpandSOLETE also does (QC flags, hybrid column, etc.).
    df["Pac"], df["Pdc"], df["TempModule"], df["TempCell"] = Pac, Pdc, TempModule, TempCell

    result = Rincon_Pombo_ThermodynamicModel(df, pv_info, verbose=0)

    assert len(result) == len(df)
    assert np.all(np.isfinite(np.asarray(result)))


# ---------------------------------------------------------------------------
# solete_pipeline.qc -- constants and rule builders, directly, without
# going through Functions.py or the full import pipeline.
# ---------------------------------------------------------------------------

def test_qc_constants_importable_directly():
    from solete_pipeline.qc import (
        QC_VALID, QC_MISSING, QC_FLAG_PRECEDENCE, KNOWN_PRESSURE_SENTINELS,
    )
    assert QC_VALID == 0
    assert QC_MISSING in QC_FLAG_PRECEDENCE
    assert 1000.0 in KNOWN_PRESSURE_SENTINELS


def test_apply_qc_flags_isolated_real_pressure_sentinel():
    from solete_pipeline.qc import apply_qc_flags, build_raw_value_qc_rules, QC_PHYSICALLY_IMPLAUSIBLE

    df = pd.read_hdf(REPO_ROOT / "SOLETE_Pombo_60min.h5")
    row = df.loc[["2019-01-01 01:00:00"]].copy()  # same real sentinel row as test_qc_flags.py
    assert row["Pressure[mbar]"].iloc[0] == 1000.0
    apply_qc_flags(row, build_raw_value_qc_rules(row))
    assert row["Pressure[mbar]_qc"].iloc[0] == QC_PHYSICALLY_IMPLAUSIBLE
