# -*- coding: utf-8 -*-
"""
solete/qc_codes.py -- the ONE quality-flag vocabulary of the SOLETE project.

Imported by the dataset pipeline (dataset/pipeline/qc_flags.py, resample_solete.py)
and by the platform (solete/qc.py, solete/expansion.py). Nothing else in the
repository may define a flag number. Pure Python + NumPy: no pandas, keras, CoolProp.

Ownership (who is allowed to EMIT a code):

    pipeline-owned   0-5, 7-10   written by dataset/pipeline/*.py from the raw sensor stream.
    platform-owned   6           QC_MODEL_SUBSTITUTED, written only by solete.expansion.
                                 It needs the PV model, so it is a property of a
                                 resolution's cleaned inputs, not of the raw stream.
                                 No dataset-pipeline rule may emit it, and it is never
                                 carried upward by resampling (Part C of the design).
    legacy-platform  11          QC_UNTREATED_IMPLAUSIBLE, written only by
                                 solete.qc.legacy_v3_raw_value_rules, i.e. for the old
                                 v3 files that were never cleaned by the pipeline.
                                 A v4 file never contains it.

Severity order (highest first) is used (a) to collapse a bucket to `_qc_worst` when
resampling and (b) to pick the flag a derived column (P_hybrid[kW]) inherits from its
constituents. Each cell holds exactly one code (not a bitmask).
"""
import numpy as np

QC_OK = 0                          # untouched, no issue detected
QC_WRAPPED = 1                     # WIND_DIR only: value was mod-360 wrapped
QC_PLACEHOLDER = 2                 # sentinel/flatline value replaced with NaN
QC_DROPOUT_SHORT_FIXED = 3         # WS/HUM joint-zero dropout, short run, interpolated
QC_DROPOUT_LONG_UNTREATED = 4      # same, long run -- value kept as recorded
QC_GLITCH_SHORT_FIXED = 5          # out-of-bound value(s), short run, interpolated
QC_MODEL_SUBSTITUTED = 6           # PLATFORM-OWNED: measured P_Solar replaced by the PV model (Pac >= 1.5*P_Solar)
QC_UNVERIFIED_PROVENANCE = 7       # value plausible and unchanged, cause not established
QC_RECOMPUTED = 8                  # RESERVED, not emitted by any v4 column (decision D1: the constant-8 Azimuth/Elevation
                                   # flag columns were dropped). Kept defined so earlier development files still load; never reuse the number.
QC_GLITCH_LONG_UNTREATED_NAN = 9   # out-of-bound, run too long to fix -- set to NaN
QC_ACTIVE_DAY = 10                 # confirmed-good value (currently: P_Gaia on two days)
QC_UNTREATED_IMPLAUSIBLE = 11      # LEGACY v3 ONLY: implausible/placeholder value detected, kept as recorded

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
    QC_UNTREATED_IMPLAUSIBLE: "implausible_value_kept_legacy_v3",
}

# Codes the dataset pipeline may emit. Anything else in a pipeline-owned `<col>_qc`
# column of a 1-second input is foreign and is neutralised by the resampler.
PIPELINE_OWNED_CODES = (
    QC_OK, QC_WRAPPED, QC_PLACEHOLDER, QC_DROPOUT_SHORT_FIXED, QC_DROPOUT_LONG_UNTREATED,
    QC_GLITCH_SHORT_FIXED, QC_UNVERIFIED_PROVENANCE, QC_RECOMPUTED,
    QC_GLITCH_LONG_UNTREATED_NAN, QC_ACTIVE_DAY,
)
PLATFORM_OWNED_CODES = (QC_MODEL_SUBSTITUTED,)

# `P_hybrid[kW]_qc_source` (int8): which constituent the hybrid flag was taken from.
SOURCE_NONE = 0     # hybrid flag is QC_OK: nothing to attribute
SOURCE_SOLAR = 1    # the flag comes from P_Solar[kW]_qc (ties also go here)
SOURCE_WIND = 2     # the flag comes from P_Gaia[kW]_qc
SOURCE_LABELS = {SOURCE_NONE: "none", SOURCE_SOLAR: "P_Solar[kW]", SOURCE_WIND: "P_Gaia[kW]"}
LEGACY_PLATFORM_CODES = (QC_UNTREATED_IMPLAUSIBLE,)

# Severity precedence, highest severity first. Rationale (unchanged from the dataset schema):
# a value now missing is a stronger claim than a value kept but of unresolved reliability,
# which is stronger than a trustworthy non-measurement (model output), which is stronger
# than a routine interpolation or a trivial correction; ACTIVE_DAY is informational.
QC_SEVERITY_ORDER = (
    QC_GLITCH_LONG_UNTREATED_NAN,   # data now missing (physically-impossible run, unfixable)
    QC_PLACEHOLDER,                 # data now missing (sentinel run, unfixable)
    QC_DROPOUT_LONG_UNTREATED,      # value kept but reliability unresolved (long dropout)
    QC_UNTREATED_IMPLAUSIBLE,       # legacy v3: implausible value kept (same claim strength as above)
    QC_MODEL_SUBSTITUTED,           # measured value replaced by a model estimate (platform-owned)
    QC_UNVERIFIED_PROVENANCE,       # value kept, cause of the reading not established
    QC_RECOMPUTED,                  # not a measurement at all, but trustworthy (model output)
    QC_GLITCH_SHORT_FIXED,          # interpolated over a short run
    QC_DROPOUT_SHORT_FIXED,         # interpolated over a short run
    QC_WRAPPED,                     # trivial, fully-determined correction
    QC_ACTIVE_DAY,                  # confirmed good; informational only
    QC_OK,                          # untouched, no issue
)
assert len(set(QC_SEVERITY_ORDER)) == len(QC_SEVERITY_ORDER) == len(QC_LABELS)

SEVERITY_RANK = {code: i for i, code in enumerate(QC_SEVERITY_ORDER)}   # 0 = worst

# Columns whose values are model-derived: computed per resolution by
# solete.expansion.expand_physical and NEVER resampled from a finer resolution.
# (Each one's provenance is also listed in dataset/docs/DATA_DICTIONARY.md.)
MODEL_DERIVED_COLUMNS = (
    "Pac", "Pdc", "TempModule", "TempCell",
    "P_Solar_clean[kW]", "P_hybrid[kW]",
    "P_Solar[kW]_qc", "P_hybrid[kW]_qc", "P_hybrid[kW]_qc_source",
)

_LUT_SIZE = 256


def _rank_lut():
    """int16 lookup table code -> severity rank; unknown codes get len(order) (weaker than everything)."""
    lut = np.full(_LUT_SIZE, len(QC_SEVERITY_ORDER), dtype=np.int16)
    for code, rank in SEVERITY_RANK.items():
        lut[code] = rank
    return lut


_RANK_LUT = _rank_lut()


def severity_rank(codes):
    """Vectorised severity rank of an integer array of codes (0 = worst). Pure NumPy."""
    c = np.asarray(codes)
    return _RANK_LUT[c.astype(np.int64, copy=False)]


def worse_of(codes_a, codes_b):
    """Element-wise: the code with the higher severity; ties return `codes_a`."""
    a = np.asarray(codes_a)
    b = np.asarray(codes_b)
    return np.where(severity_rank(b) < severity_rank(a), b, a)
