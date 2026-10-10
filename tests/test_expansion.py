# -*- coding: utf-8 -*-
"""
Tests for solete/expansion.py (Part B/C of the QC + expansion task) and the single QC vocabulary.

What is proven here:
  * the NumPy PV model is bit-for-bit the pre-refactor one (a frozen copy is embedded below),
  * expand_physical is row-wise: chunking, row order and single-row evaluation all give identical numbers,
  * it is idempotent on a file that already carries its model columns (incl. a code 6),
  * the measured P_Solar[kW] is never touched; P_Solar_clean[kW] equals what the old in-place overwrite
    produced, and ExpandSOLETE's working P_Solar[kW] equals P_Solar_clean[kW],
  * one QC vocabulary, code 6 only from the platform, never resampled upward,
  * data_version='v4' loads and never overwrites the file's flags.
Real-data cases use the v3 hourly file and SKIP when it is absent; everything else runs on the tracked
sample files or on solete/synthetic.py (synthetic, labelled as such).
"""
import sys
import types
import pathlib
import inspect

for _modname in ["keras", "keras.models", "keras.layers"]:     # same stub as test_qc_flags.py
    sys.modules.setdefault(_modname, types.ModuleType(_modname))
sys.modules["keras.models"].Sequential = object
sys.modules["keras.models"].load_model = lambda *a, **k: None
for _n in ["LSTM", "Dense", "Masking", "Flatten", "Conv1D", "MaxPooling1D"]:
    setattr(sys.modules["keras.layers"], _n, object)

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _data import short_sample_path, v3_60min_path
from solete import expansion, paths, qc, qc_codes
from solete.qc_codes import SOURCE_LABELS, SOURCE_NONE, SOURCE_SOLAR, SOURCE_WIND
from solete.expansion import (compute_physical, expand_physical, PHYSICAL_COLUMNS, SOLAR_QC, HYBRID_QC, HYBRID_SOURCE)
from solete.io import import_SOLETE_data, import_SOLETE_sample, import_PV_WT_data
from solete.physics import PV_Performance_Model, pv_model_arrays
from solete.preprocessing import ExpandSOLETE
from solete.synthetic import synthetic_solete

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PV, WT = import_PV_WT_data()
SAMPLE = REPO_ROOT / "examples" / "SOLETE_sample.h5"


# ---------------------------------------------------------------------------
# Frozen reference implementations (the code as it was before the refactor)
# ---------------------------------------------------------------------------

def reference_PV_Performance_Model(data, PVinfo, colirra='POA Irr[kW1m2]', coltemp='TEMPERATURE[degC]',colwindspeed='WIND_SPEED[m1s]'):
    """Frozen copy of the pre-refactor implementation (DataFrame based), used only as a reference."""
    
    
    # Obtains the expected solar production based on irradiance, temperature, pv parameters, etc
    DATA_PV = pd.DataFrame({'Pmp_stc' : PVinfo["Pmp_stc"],
                            'ganma_mp' : PVinfo['ganma_mp'],
                            'Ns': PVinfo['Ns'],
                            'Np': PVinfo['Np'],
                            'a' : PVinfo['a'],
                            'b' : PVinfo['b'],
                            'D_T' : PVinfo['D_T'],
                            'eff_P' : PVinfo['eff_P'],
                            'eff_%' : PVinfo['eff_%'],
                            }, 
                           index = PVinfo["index"])
    
    DATA_PV['eff_max_%'] = [max(DATA_PV['eff_%'].loc['A']), max(DATA_PV['eff_%'].loc['B'])] #maximum inverter efficiency in %
    DATA_PV['eff_max_P'] = [max(DATA_PV['eff_P'].loc['A']), max(DATA_PV['eff_P'].loc['B'])] #W maximum power output of the inverter
    
    Results = pd.DataFrame(index = data.index)
    
    for pv in DATA_PV.index:
        #Temperature Module
        Results['Tm_' + pv] = data[coltemp] + data[colirra]*1000 *np.exp(DATA_PV.loc[pv,'a']+DATA_PV.loc[pv,'b']*data[colwindspeed]) 
        #Temperature Cell
        Results['Tc_' + pv] = Results['Tm_' + pv] + data[colirra]*1000/PVinfo["Estc"] * DATA_PV.loc[pv,'D_T']
        #power produced in one single pannel
        Results['Pmp_panel_' + pv] = data[colirra]*1000/PVinfo["Estc"] * DATA_PV.loc[pv, 'Pmp_stc'] * (1+DATA_PV.loc[pv, 'ganma_mp'] * (Results['Tc_' + pv] - PVinfo["Tstc"]) )
        #power produced by all the panels in the array
        Results['Pmp_array_' + pv] = DATA_PV.loc[pv, 'Ns'] * DATA_PV.loc[pv, 'Np'] * Results['Pmp_panel_' + pv]
        #efficiency of the inverter corresponding to the instantaneous power output
        Results['eff_inv_' + pv] =  np.interp(Results['Pmp_array_' + pv], DATA_PV.loc[pv, 'eff_P'], DATA_PV.loc[pv, 'eff_%'], left=0)/100
        
        
        Results['Pac_' + pv] =  DATA_PV.loc[pv, 'eff_max_%']/100 * Results['Pmp_array_' + pv]
        #If any of the Pac is > than the maximum capacity of the inverter, then use the max capacity of the inverter.
        #NOTE: this must only touch the Pac_<pv> column -- Results[mask]=value (without .loc[mask, col]) applies
        #the scalar to every column of Results for the masked rows, silently clobbering Tm/Tc/Pmp_panel/Pmp_array/eff_inv too.
        Results.loc[Results['Pac_' + pv]>DATA_PV.loc[pv, 'eff_max_P'], 'Pac_' + pv]=DATA_PV.loc[pv, 'eff_max_P']
        Results.loc[Results['Pac_' + pv]<0, 'Pac_' + pv]=0
        
    return Results[['Pac_A', 'Pac_B']].sum(axis=1)/1000, Results[['Pmp_array_A', 'Pmp_array_B']].sum(axis=1)/1000, Results[['Tm_A', 'Tm_B']].mean(axis=1), Results[['Tc_A', 'Tc_B']].mean(axis=1)


def reference_old_expansion(df, pv):
    """The numeric steps the old ExpandSOLETE performed in place, on a copy. Returns a dict."""
    d = df.copy()
    Pac, Pdc, Tm, Tc = reference_PV_Performance_Model(d, pv)
    subst_old = Pac >= 1.5 * d['P_Solar[kW]']                  # the old rule (true for 0 vs 0)
    subst = subst_old & (np.where(Pac <= 0.001, 0, Pac) > 0)    # the current rule: and the model is producing
    P = np.where(subst_old, Pac, d['P_Solar[kW]'])              # the OLD substitution: values are what the old code produced
    P = np.where(P <= 0.001, 0, P)
    return {'Pac': np.where(Pac <= 0.001, 0, Pac), 'Pdc': Pdc.to_numpy(), 'TempModule': Tm.to_numpy(),
            'TempCell': Tc.to_numpy(), 'subst': subst.to_numpy(),
            'subst_old': subst_old.to_numpy(), 'P_Solar': P,
            'P_hybrid[kW]': P + d['P_Gaia[kW]'].to_numpy()}


def _bits_equal(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


@pytest.fixture(scope="module")
def short():
    return pd.read_hdf(short_sample_path())


@pytest.fixture(scope="module")
def sample():
    return pd.read_hdf(SAMPLE)


@pytest.fixture(scope="module")
def syn():
    """Synthetic 2 days of 1 s data with NaNs injected in the inputs, an inverter-clamp row and a negative P_Solar."""
    df = synthetic_solete('2019-06-01', periods=2 * 86400, seed=3)
    rng = np.random.default_rng(1)
    for c in ['POA Irr[kW1m2]', 'TEMPERATURE[degC]', 'WIND_SPEED[m1s]', 'P_Solar[kW]']:
        df.loc[rng.random(len(df)) < 0.003, c] = np.nan
    df.iloc[40000, df.columns.get_loc('POA Irr[kW1m2]')] = 1e5       # Pac clamped at the inverter maximum
    df.iloc[40001, df.columns.get_loc('P_Solar[kW]')] = -0.01        # negative measurement
    return df


# ---------------------------------------------------------------------------
# PV model: NumPy version == frozen old version, bit for bit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["short", "sample", "syn"])
def test_pv_model_bit_identical_to_pre_refactor(which, short, sample, syn):
    df = {"short": short, "sample": sample, "syn": syn}[which]
    old = reference_PV_Performance_Model(df, PV)
    new = PV_Performance_Model(df, PV)
    for o, n in zip(old, new):
        assert _bits_equal(o.to_numpy(), n.to_numpy())


def test_pv_model_bit_identical_real_hourly_file():
    df = pd.read_hdf(v3_60min_path())
    for o, n in zip(reference_PV_Performance_Model(df, PV), PV_Performance_Model(df, PV)):
        assert _bits_equal(o.to_numpy(), n.to_numpy())


def test_pv_model_nan_input_gives_zero_power_as_before(syn):
    Pac, Pdc, Tm, Tc = pv_model_arrays(syn['POA Irr[kW1m2]'].to_numpy(), syn['TEMPERATURE[degC]'].to_numpy(),
                                       syn['WIND_SPEED[m1s]'].to_numpy(), PV)
    bad = syn[['POA Irr[kW1m2]', 'TEMPERATURE[degC]', 'WIND_SPEED[m1s]']].isna().any(axis=1).to_numpy()
    assert bad.any() and (Pac[bad] == 0).all() and np.isnan(Tm[bad]).all()


# ---------------------------------------------------------------------------
# expand_physical == the old in-place numbers; measured P_Solar untouched
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["short", "sample", "syn"])
def test_expand_physical_matches_old_expansion_and_keeps_measurement(which, short, sample, syn):
    df = {"short": short, "sample": sample, "syn": syn}[which].copy()
    before = df.copy()
    ref = reference_old_expansion(df, PV)
    expand_physical(df, PV)
    for c in before.columns:                                            # inputs untouched, incl. measured P_Solar
        pd.testing.assert_series_equal(df[c], before[c])
    for c in ['Pac', 'Pdc', 'TempModule', 'TempCell', 'P_hybrid[kW]']:
        assert _bits_equal(df[c], ref[c]), c
    assert _bits_equal(df['P_Solar_clean[kW]'], ref['P_Solar'])         # == what the old overwrite produced
    assert ((df[SOLAR_QC].to_numpy() == 6) == ref['subst']).all()


def test_expand_physical_real_hourly_file_equals_old_overwrite():
    df = pd.read_hdf(v3_60min_path())
    ref = reference_old_expansion(df, PV)
    measured = df['P_Solar[kW]'].copy()
    expand_physical(df, PV)
    assert _bits_equal(df['P_Solar_clean[kW]'], ref['P_Solar'])
    pd.testing.assert_series_equal(df['P_Solar[kW]'], measured)
    # the old rule flagged 4,204 rows (38.33 %), every one a night row with measured 0 and model 0;
    # with "and Pac > 0" none is flagged, and not a single value changes (P_Solar_clean checked above)
    assert int(ref['subst_old'].sum()) == 4204
    assert int((df[SOLAR_QC] == 6).sum()) == int(ref['subst'].sum()) == 0


def test_expandsolete_working_solar_equals_clean_and_old_numbers(short):
    df = short.copy()
    ref = reference_old_expansion(df, PV)
    cv = {"OriginalFeatures": list(df.columns), "PossibleFeatures": []}
    ExpandSOLETE(df, [PV, WT], cv)
    assert _bits_equal(df['P_Solar[kW]'], ref['P_Solar'])
    assert _bits_equal(df['P_Solar[kW]'], df['P_Solar_clean[kW]'])
    assert _bits_equal(df['P_hybrid[kW]'], ref['P_hybrid[kW]'])


def test_expandsolete_ml_features_stay_in_expandsolete(short):
    df = short.copy()
    cv = {"OriginalFeatures": list(df.columns), "IntrinsicFeature": "P_Solar[kW]", "H": 3,
          "PossibleFeatures": ["HoursOfDay", "MeanPrevH", "StdPrevH", "MeanWindSpeedPrevH", "StdWindSpeedPrevH"]}
    ExpandSOLETE(df, [PV, WT], cv)
    for c in cv["PossibleFeatures"]:
        assert c in df.columns
    out = expand_physical(short.copy(), PV)
    assert not any(c in out.columns for c in cv["PossibleFeatures"] + ["TempModule_RP"])
    assert HYBRID_SOURCE in df.columns                                    # ExpandSOLETE keeps the label for the platform frame


# ---------------------------------------------------------------------------
# Row-wise, resolution-agnostic: chunking, order, single rows
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chunk", [1, 7, 1000, 12345])
def test_chunked_equals_full_run(syn, chunk):
    small = syn.iloc[:60000]
    full = compute_physical(small, PV, chunk_rows=None)
    chunked = compute_physical(small, PV, chunk_rows=chunk)
    pd.testing.assert_frame_equal(full, chunked, check_exact=True)
    for c in full.columns:
        if full[c].dtype.kind == 'f':
            assert _bits_equal(full[c], chunked[c])


def test_slice_equals_same_rows_of_full_run(syn):
    full = compute_physical(syn.iloc[:50000], PV, chunk_rows=None)
    part = compute_physical(syn.iloc[12345:23456], PV, chunk_rows=3000)
    pd.testing.assert_frame_equal(full.iloc[12345:23456], part, check_exact=True)


def test_row_order_and_single_rows_do_not_matter(short):
    perm = np.random.default_rng(0).permutation(len(short))
    a = compute_physical(short, PV)
    b = compute_physical(short.iloc[perm], PV)
    pd.testing.assert_frame_equal(a.iloc[perm], b, check_exact=True)
    for i in (0, 11, 23):
        one = compute_physical(short.iloc[[i]], PV)
        pd.testing.assert_frame_equal(a.iloc[[i]], one, check_exact=True)


def test_duplicate_and_unsorted_index_is_fine(short):
    df = pd.concat([short, short.iloc[::-1]])          # duplicate, non-monotonic timestamps (as in the v3 file)
    out = expand_physical(df.copy(), PV, chunk_rows=10)
    assert len(out) == 48 and out['Pac'].iloc[:24].equals(out['Pac'].iloc[24:][::-1].set_axis(out.index[:24]))


def test_no_vectorize_no_rolling_no_per_row_loops():
    for obj in (expansion, qc, qc_codes):
        text = inspect.getsource(obj)
        assert 'np.vectorize' not in text and '.rolling(' not in text and 'iterrows' not in text and 'apply(' not in text
    assert 'np.vectorize' not in inspect.getsource(sys.modules['solete.preprocessing'].ExpandSOLETE)


def test_night_rows_are_not_substitutions():
    night = pd.DataFrame({'TEMPERATURE[degC]': [5.0, 5.0, 5.0], 'WIND_SPEED[m1s]': [2.0] * 3, 'POA Irr[kW1m2]': [0.0, 0.0, 0.0],
                          'P_Gaia[kW]': [0.0] * 3, 'P_Solar[kW]': [0.0, -0.01, 0.0004]})
    out = expand_physical(night, PV)
    assert (out[SOLAR_QC] == 0).all()
    assert (out['P_Solar_clean[kW]'] == 0).all()                     # values: zeros either way
    day = pd.DataFrame({'TEMPERATURE[degC]': [15.0], 'WIND_SPEED[m1s]': [3.0], 'POA Irr[kW1m2]': [0.8],
                        'P_Gaia[kW]': [0.0], 'P_Solar[kW]': [0.0]})   # sun up, measured 0 -> genuine substitution
    d = expand_physical(day, PV)
    assert d[SOLAR_QC].iloc[0] == 6 and d['P_Solar_clean[kW]'].iloc[0] > 1


def test_source_is_an_int8_code_with_documented_labels(short):
    out = compute_physical(short, PV)
    assert out[HYBRID_SOURCE].dtype == np.int8
    assert set(out[HYBRID_SOURCE].unique()) <= set(SOURCE_LABELS)
    assert 'P_Solar_model_substituted' not in out.columns             # the boolean was identical to `P_Solar[kW]_qc == 6`
    assert set(out.columns) == set(PHYSICAL_COLUMNS)


# ---------------------------------------------------------------------------
# Idempotence on a file that already holds its model columns
# ---------------------------------------------------------------------------

def test_idempotent_on_expanded_file(sample):
    df = sample.copy()
    # a pipeline-style file: P_Gaia_qc is 7 or 10 on every row, other columns carry flags
    df['P_Gaia[kW]_qc'] = np.where(df.index.month == 1, 10, 7).astype(np.int8)
    df['Pressure[mbar]_qc'] = np.int8(2)
    sunny = (df['POA Irr[kW1m2]'] > 0.3).to_numpy()
    df.loc[sunny, 'P_Solar[kW]'] = df.loc[sunny, 'P_Solar[kW]'] * 0.3       # curtailment-like: measured far below the model
    once = expand_physical(df.copy(), PV)
    assert (once[SOLAR_QC] == qc_codes.QC_MODEL_SUBSTITUTED).any()       # a code 6 is present now
    twice = expand_physical(once.copy(), PV)
    pd.testing.assert_frame_equal(once, twice, check_exact=True)
    thrice = expand_physical(twice.copy(), PV, chunk_rows=500)            # also through another chunking
    pd.testing.assert_frame_equal(once, thrice, check_exact=True)


def test_idempotent_on_synthetic_with_nans(syn):
    s = syn.iloc[:30000]
    once = expand_physical(s.copy(), PV)
    pd.testing.assert_frame_equal(once, expand_physical(once.copy(), PV, chunk_rows=999), check_exact=True)


# ---------------------------------------------------------------------------
# QC: one vocabulary, severity inheritance, code 6 ownership
# ---------------------------------------------------------------------------

def test_vocabulary_is_single_and_consistent():
    pipeline = pathlib.Path(REPO_ROOT / "dataset" / "pipeline")
    sys.path.insert(0, str(pipeline))
    try:
        import qc_flags
        for name in ["QC_OK", "QC_WRAPPED", "QC_PLACEHOLDER", "QC_DROPOUT_SHORT_FIXED", "QC_DROPOUT_LONG_UNTREATED",
                     "QC_GLITCH_SHORT_FIXED", "QC_UNVERIFIED_PROVENANCE", "QC_RECOMPUTED",
                     "QC_GLITCH_LONG_UNTREATED_NAN", "QC_ACTIVE_DAY", "QC_MODEL_SUBSTITUTED"]:
            assert getattr(qc_flags, name) == getattr(qc_codes, name)
        assert qc_flags.QC_SEVERITY_ORDER is qc_codes.QC_SEVERITY_ORDER
        assert qc_flags.QC_LABELS is qc_codes.QC_LABELS
        # no pipeline rule may emit 6 (or the legacy 11)
        qc_flags.assert_pipeline_codes(np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10]))
        for bad in (6, 11, 99):
            with pytest.raises(ValueError):
                qc_flags.assert_pipeline_codes(np.array([0, bad]))
    finally:
        sys.path.remove(str(pipeline))
    assert qc_codes.QC_MODEL_SUBSTITUTED == 6
    assert set(qc_codes.QC_SEVERITY_ORDER) == set(qc_codes.QC_LABELS) == set(range(12))
    assert 6 not in qc_codes.PIPELINE_OWNED_CODES and 6 in qc_codes.PLATFORM_OWNED_CODES


def test_no_pipeline_script_defines_a_flag_number():
    text = (REPO_ROOT / "dataset" / "pipeline" / "qc_flags.py").read_text()
    assert "QC_OK = 0" not in text and "from solete.qc_codes import" in text


def test_substitution_flag_keeps_more_severe_codes_and_ignores_stale_six():
    existing = np.array([0, 6, 9, 2, 10, 7], dtype=np.int8)
    out = qc.add_substitution_flag(existing, np.array([True, False, True, True, True, False]))
    # 0 -> 6 ; stale 6 with no substitution -> 0 ; 9 and 2 outrank 6 and stay ; 10 -> 6 ; 7 stays
    assert out.tolist() == [6, 0, 9, 2, 6, 7]
    assert out.dtype == np.int8


def test_hybrid_flag_uses_single_severity_order():
    df = pd.DataFrame({
        'TEMPERATURE[degC]': [15.0] * 6, 'WIND_SPEED[m1s]': [3.0] * 6, 'POA Irr[kW1m2]': [0.8] * 6,
        'P_Gaia[kW]': [0.0] * 6, 'P_Solar[kW]': [0.0] * 6,                    # sun up, measured 0: substitution fires
        'P_Solar[kW]_qc': np.array([0, 0, 9, 0, 0, 0], dtype=np.int8),
        'P_Gaia[kW]_qc': np.array([7, 10, 7, 0, 7, 9], dtype=np.int8)})
    out = expand_physical(df.copy(), PV)
    # solar: [6,6,9,6,6,6]  wind: [7,10,7,0,7,9]
    assert out[SOLAR_QC].tolist() == [6, 6, 9, 6, 6, 6]
    assert out[HYBRID_QC].tolist() == [6, 6, 9, 6, 6, 9]       # 6 outranks 7 and 10; 9 outranks 6
    assert out[HYBRID_SOURCE].tolist() == [SOURCE_SOLAR] * 5 + [SOURCE_WIND]


def test_hybrid_flag_uses_qc_worst_of_resampled_files():
    df = pd.DataFrame({'TEMPERATURE[degC]': [15.0] * 3, 'WIND_SPEED[m1s]': [3.0] * 3, 'POA Irr[kW1m2]': [0.8] * 3,
                       'P_Gaia[kW]': [0.0] * 3, 'P_Solar[kW]': pv_model_arrays(np.array([0.8]), np.array([15.0]),
                                                                          np.array([3.0]), PV)[0].repeat(3) * 0.97,
                       'P_Gaia[kW]_qc_worst': [7.0, 10.0, np.nan]})
    out = expand_physical(df, PV)
    assert out[HYBRID_QC].tolist() == [7, 10, 0]
    assert out[HYBRID_SOURCE].tolist() == [SOURCE_WIND, SOURCE_WIND, SOURCE_NONE]


def test_hybrid_source_none_when_both_ok():
    pac = pv_model_arrays(np.array([0.8]), np.array([15.0]), np.array([3.0]), PV)[0]
    df = pd.DataFrame({'TEMPERATURE[degC]': [15.0], 'WIND_SPEED[m1s]': [3.0], 'POA Irr[kW1m2]': [0.8],
                       'P_Gaia[kW]': [0.0], 'P_Solar[kW]': pac * 0.97})     # daytime, measured close to the model
    out = expand_physical(df, PV)
    assert out[SOLAR_QC].iloc[0] == 0
    assert out[HYBRID_QC].iloc[0] == 0 and out[HYBRID_SOURCE].iloc[0] == SOURCE_NONE


def test_hybrid_flag_unchanged_on_v3_hourly_file():
    df = pd.read_hdf(v3_60min_path())
    out = expand_physical(df, PV)
    pd.testing.assert_series_equal(out[HYBRID_QC], out[SOLAR_QC], check_names=False)
    assert not (out[HYBRID_SOURCE] == SOURCE_WIND).any()


# ---------------------------------------------------------------------------
# Resampling: only measured columns and pipeline-owned flags go up
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def resampler():
    pipeline = str(REPO_ROOT / "dataset" / "pipeline")
    sys.path.insert(0, pipeline)
    try:
        import resample_solete
        yield resample_solete
    finally:
        sys.path.remove(pipeline)


def test_resampling_never_carries_code_six_or_model_columns(resampler):
    df = synthetic_solete('2019-06-01 06:00', periods=2 * 3600, seed=5)
    df.iloc[300:400, df.columns.get_loc('P_Solar[kW]')] = 0.0              # measured 0 while the sun is up -> substituted
    df['Pressure[mbar]_qc'] = np.int8(0)
    df['HUMIDITY[%]_qc'] = np.int8(0)
    df.iloc[100:130, df.columns.get_loc('Pressure[mbar]_qc')] = 2          # a genuine pipeline flag
    df.iloc[200:260, df.columns.get_loc('HUMIDITY[%]_qc')] = 6            # FOREIGN code 6 inside a pipeline flag column
    expanded = expand_physical(df.copy(), PV)                              # 1 s model columns, code 6 in P_Solar_qc
    assert (expanded[SOLAR_QC] == 6).any()
    for rule in ('1min', '5min', '1h'):
        out = resampler.resample_dataframe(expanded, rule)
        assert not set(qc_codes.MODEL_DERIVED_COLUMNS) & set(out.columns)
        worst = [c for c in out.columns if c.endswith('_qc_worst')]
        assert worst and all(not (out[c] == 6).any() for c in worst)
        assert not any(c.startswith('P_Solar[kW]_qc') or c.startswith('P_hybrid') for c in out.columns)
    one = resampler.resample_dataframe(expanded, '1min')
    assert one['Pressure[mbar]_qc_worst'].max() == 2                       # real flags still come through
    assert one['HUMIDITY[%]_qc_worst'].max() == 0                          # the foreign 6 became QC_OK
    assert one['HUMIDITY[%]_qc_frac_flagged'].max() == 0


def test_resampler_lists_are_the_shared_ones(resampler):
    assert set(resampler.SKIP_MODEL_COLUMNS) == set(qc_codes.MODEL_DERIVED_COLUMNS)
    assert 6 not in resampler.PIPELINE_OWNED_CODES


def test_model_columns_computed_per_resolution_differ_from_averaged_one_second():
    """The documented design: expand(resampled inputs) is NOT resample(expand(1 s inputs)) (synthetic data)."""
    one = synthetic_solete('2019-06-01', periods=2 * 86400, seed=2).dropna()
    hourly = one.resample('60min', label='left', closed='left').mean()
    direct = compute_physical(hourly, PV)['Pac']
    averaged = compute_physical(one, PV)['Pac'].resample('60min', label='left', closed='left').mean()
    assert not np.allclose(direct, averaged, rtol=0, atol=1e-6)
    assert abs(direct.mean() - averaged.mean()) < 0.1 * averaged.mean()    # but close in aggregate


# ---------------------------------------------------------------------------
# data_version='v4' through import_SOLETE_data
# ---------------------------------------------------------------------------

@pytest.fixture()
def v4_dir(tmp_path, monkeypatch):
    hdf5 = tmp_path / "hdf5"
    hdf5.mkdir()
    df = synthetic_solete('2019-05-25', periods=24 * 12, freq='60min', seed=4)
    df['Azimuth[deg]'], df['Elevation[deg]'] = 10.0, 20.0
    rng = np.random.default_rng(0)
    flags = {'WIND_DIR[deg]_qc': rng.choice([0, 1], len(df)), 'Pressure[mbar]_qc': rng.choice([0, 2, 9], len(df)),
             'HUMIDITY[%]_qc': rng.choice([0, 3, 4, 5], len(df)), 'P_Gaia[kW]_qc': rng.choice([7, 10], len(df)),
             'Azimuth[deg]_qc': np.full(len(df), 8), 'Elevation[deg]_qc': np.full(len(df), 8)}
    for k, v in flags.items():
        df[k] = v.astype(np.int8)
    df.to_hdf(hdf5 / "SOLETE_Pombo_60min_v4.h5", key="DATA", mode="w")
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    monkeypatch.setattr(paths, "HDF5_DIR", hdf5)
    return df


def test_v4_loads_keeps_flags_and_sets_working_solar_from_clean(v4_dir):
    cv = {"resolution": "60min", "data_version": "v4", "SOLETE_builvsimport": "Build", "SOLETE_save": False,
          "OriginalFeatures": [], "PossibleFeatures": []}
    out = import_SOLETE_data(cv, PV, WT)
    for c in [c for c in v4_dir.columns if c.endswith('_qc')]:             # pipeline flags: byte-identical, int8
        assert out[c].to_numpy().tobytes() == v4_dir[c].to_numpy().tobytes() and out[c].dtype == np.int8
    assert 11 not in set(np.unique(out[[c for c in out.columns if c.endswith('_qc')]].to_numpy()))  # no legacy rules on v4
    assert _bits_equal(out['P_Solar[kW]'], out['P_Solar_clean[kW]'])
    ref = reference_old_expansion(v4_dir, PV)
    assert _bits_equal(out['P_Solar[kW]'], ref['P_Solar'])                 # benchmark behaviour == old in-memory overwrite
    assert set(np.unique(out[SOLAR_QC])) <= {0, 6}
    assert not (out[HYBRID_QC] == 11).any()


def test_v4_unknown_flag_code_is_rejected(v4_dir, tmp_path):
    bad = v4_dir.copy()
    bad['Pressure[mbar]_qc'] = np.int8(42)
    bad.to_hdf(tmp_path / "hdf5" / "SOLETE_Pombo_60min_v4.h5", key="DATA", mode="w")
    cv = {"resolution": "60min", "data_version": "v4", "SOLETE_builvsimport": "Build", "SOLETE_save": False,
          "OriginalFeatures": [], "PossibleFeatures": []}
    with pytest.raises(ValueError, match="Unknown QC codes"):
        import_SOLETE_data(cv, PV, WT)


def test_v3_raw_value_flags_flag_the_same_rows_with_code_11(short):
    df = pd.read_hdf(SAMPLE)
    qc.apply_qc_flags(df, qc.legacy_v3_raw_value_rules(df))
    assert set(np.unique(df['Pressure[mbar]_qc'])) <= {0, 11}
    assert (df['Pressure[mbar]_qc'] == 11).sum() == (df['Pressure[mbar]'].isin({1000.0, 2000.0, 3000.0}) |
                                                    (df['Pressure[mbar]'] < 870) | (df['Pressure[mbar]'] > 1085)).sum()
