"""Canonical quality-control codes shared by the v4 dataset and platform."""

QC_OK = 0
QC_WRAPPED = 1
QC_PLACEHOLDER = 2
QC_DROPOUT_SHORT_FIXED = 3
QC_DROPOUT_LONG_UNTREATED = 4
QC_GLITCH_SHORT_FIXED = 5
QC_MODEL_SUBSTITUTED = 6
QC_UNVERIFIED_PROVENANCE = 7
QC_RECOMPUTED = 8
QC_GLITCH_LONG_UNTREATED_NAN = 9
QC_ACTIVE_DAY = 10

QC_LABELS = {
    QC_OK: "ok",
    QC_WRAPPED: "wrapped_mod_360",
    QC_PLACEHOLDER: "placeholder_set_to_nan",
    QC_DROPOUT_SHORT_FIXED: "dropout_short_run_interpolated",
    QC_DROPOUT_LONG_UNTREATED: "dropout_long_run_untreated",
    QC_GLITCH_SHORT_FIXED: "glitch_short_run_interpolated",
    QC_MODEL_SUBSTITUTED: "model_substituted",
    QC_UNVERIFIED_PROVENANCE: "unverified_provenance",
    QC_RECOMPUTED: "recomputed_replacing_measurement",
    QC_GLITCH_LONG_UNTREATED_NAN: "glitch_long_run_untreated_set_to_nan",
    QC_ACTIVE_DAY: "p_gaia_confirmed_active_day",
}

QC_SEVERITY_ORDER = (
    QC_GLITCH_LONG_UNTREATED_NAN,
    QC_PLACEHOLDER,
    QC_DROPOUT_LONG_UNTREATED,
    QC_MODEL_SUBSTITUTED,
    QC_UNVERIFIED_PROVENANCE,
    QC_RECOMPUTED,
    QC_GLITCH_SHORT_FIXED,
    QC_DROPOUT_SHORT_FIXED,
    QC_WRAPPED,
    QC_ACTIVE_DAY,
    QC_OK,
)

PIPELINE_QC_CODES = frozenset({
    QC_OK,
    QC_WRAPPED,
    QC_PLACEHOLDER,
    QC_DROPOUT_SHORT_FIXED,
    QC_DROPOUT_LONG_UNTREATED,
    QC_GLITCH_SHORT_FIXED,
    QC_UNVERIFIED_PROVENANCE,
    QC_RECOMPUTED,
    QC_GLITCH_LONG_UNTREATED_NAN,
    QC_ACTIVE_DAY,
})

PIPELINE_QC_SOURCE_COLUMNS = (
    "WIND_DIR[deg]",
    "Pressure[mbar]",
    "WIND_SPEED[m1s]",
    "HUMIDITY[%]",
    "TEMPERATURE[degC]",
    "GHI[kW1m2]",
    "POA Irr[kW1m2]",
    "P_Gaia[kW]",
    "Azimuth[deg]",
    "Elevation[deg]",
)
PIPELINE_QC_COLUMNS = frozenset(f"{column}_qc" for column in PIPELINE_QC_SOURCE_COLUMNS)

MODEL_DERIVED_COLUMNS = frozenset({
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
})


def qc_rank(values):
    """Return severity ranks for scalar or array-like QC code values."""
    rank_by_code = {code: rank for rank, code in enumerate(QC_SEVERITY_ORDER)}
    return values.map(rank_by_code) if hasattr(values, "map") else rank_by_code.get(values, len(rank_by_code))