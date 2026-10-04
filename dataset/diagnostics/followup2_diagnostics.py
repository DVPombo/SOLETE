"""
followup2_diagnostics.py -- sections F and G.  READ-ONLY.

!! REWRITTEN, NOT THE ORIGINAL. The original followup2_diagnostics.py written in a
!! prior session was not present in the SOLETEdataset zip handed over for this
!! session (only followup_diagnostics.py, sections A-E, was). This file is a fresh
!! implementation of the two questions the handoff describes; if the original
!! turns up, prefer it, or diff the report names/fields against this one.

QUESTION: were the ORIGINAL published hourly file's TEMPERATURE / HUMIDITY /
WIND_SPEED / Pressure columns built from a differently-adjusted raw version
rather than a plain hourly mean of the raw 1-second file -- the way WIND_DIR
turned out to be (hourly values above the raw hour's own maximum; see
CLEANING_DECISIONS.md section 3)?

  followup2_F_mismatch_hours
      Per column: plain hourly mean of the RAW (sorted) 1-second file vs the
      original hourly value. Label convention is chosen per column from
      shifts -1/0/+1 (bucket labelled by start, [T, T+1h)). Counts hours that
      do not match, and -- the decisive test -- how many mismatching hours have
      an original value OUTSIDE that hour's own raw [min, max]: no averaging
      of that hour's raw samples can produce such a value, so those hours must
      trace to different raw data. Also tests whether the original matches a
      different aggregator (median / first / last) instead of the mean.

  followup2_G_explained_by_cleaning      (only if --clean is given)
      For the mismatching hours: does the original instead match the hourly
      mean of the CLEANED 1-second file (NaN-skipping)? That would mean the
      original was built after a similar adjustment (sentinels removed, glitches
      interpolated) rather than from the raw stream.

Usage:
    python followup2_diagnostics.py SOLETE_Pombo_1sec.h5 SOLETE_Pombo_60min.h5 \
        --clean SOLETE_Pombo_1sec_clean.h5
    # Jupyter: !python3 followup2_diagnostics.py ... > f2.txt 2>&1

Memory: only the four columns of interest are loaded from each 1-second file.
NOTE: not yet run against the real files. Be skeptical of a single run: compare the
verdict_hint per column with the raw numbers, and re-run with --tol if the match
fraction sits near a threshold.
"""
import argparse
import traceback

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "pipeline"))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root: gives `import solete.paths`
from solete.paths import resolve_input
from solete_report import print_report

COLS = ["TEMPERATURE[degC]", "HUMIDITY[%]", "WIND_SPEED[m1s]", "Pressure[mbar]"]
SHIFTS = (-1, 0, 1)


def load_cols(path, key, cols):
    try:
        df = pd.read_hdf(path, key=key, columns=cols)   # table format
    except (TypeError, ValueError, NotImplementedError):
        df = pd.read_hdf(path, key=key)[cols]            # fixed format: no column subset
    return df


def sorted_by_index(df):
    order = np.argsort(df.index.values, kind="stable")
    out = pd.DataFrame({c: df[c].to_numpy()[order] for c in df.columns},
                       index=pd.DatetimeIndex(df.index.values[order]))
    return out


def hourly(df, how):
    r = df.resample("1h", label="left", closed="left")
    return getattr(r, how)()


def best_shift(fresh, orig, tol_rel, tol_abs):
    best = None
    for s in SHIFTS:
        a = pd.concat([fresh.shift(s), orig], axis=1, join="inner").dropna()
        if a.empty:
            continue
        d = (a.iloc[:, 0] - a.iloc[:, 1]).abs()
        score = float(d.median())
        if best is None or score < best[1]:
            best = (s, score)
    return best[0] if best else 0


def match(a, b, rel, ab):
    return np.isclose(a, b, rtol=rel, atol=ab)


def section_F(raw, orig, tol_rel, tol_abs):
    means, mins, maxs = hourly(raw, "mean"), hourly(raw, "min"), hourly(raw, "max")
    medians, firsts, lasts = hourly(raw, "median"), hourly(raw, "first"), hourly(raw, "last")
    out, mismatch_hours = {}, {}
    for col in COLS:
        if col not in orig.columns:
            out[col] = {"error": "column missing in original hourly file"}
            continue
        sh = best_shift(means[col], orig[col], tol_rel, tol_abs)
        sh_idx = lambda s: s.shift(sh)  # noqa: E731
        df = pd.DataFrame({"orig": orig[col], "mean": sh_idx(means[col]), "min": sh_idx(mins[col]),
                           "max": sh_idx(maxs[col]), "median": sh_idx(medians[col]),
                           "first": sh_idx(firsts[col]), "last": sh_idx(lasts[col])}).dropna(subset=["orig", "mean"])
        ok = match(df["orig"], df["mean"], tol_rel, tol_abs)
        mm = df[~ok]
        diff = (mm["orig"] - mm["mean"]).abs()
        outside = (mm["orig"] < mm["min"] - tol_abs) | (mm["orig"] > mm["max"] + tol_abs)
        alt = {k: float(match(mm["orig"], mm[k], tol_rel, tol_abs).mean()) if len(mm) else None
               for k in ("median", "first", "last")}
        mismatch_hours[col] = mm.index
        res = {
            "label_shift_used_periods": sh, "n_hours_compared": int(len(df)),
            "n_match_plain_mean": int(ok.sum()), "n_mismatch": int(len(mm)),
            "pct_mismatch": float(100 * len(mm) / max(1, len(df))),
            "tolerance": {"rtol": tol_rel, "atol": tol_abs},
            "mismatch_abs_diff": ({"median": float(diff.median()), "p95": float(diff.quantile(.95)),
                                   "max": float(diff.max())} if len(mm) else None),
            "n_mismatch_outside_raw_hour_range": int(outside.sum()),
            "frac_mismatch_matching_other_aggregator": alt,
            "mismatch_days_top10": {str(k.date()): int(v) for k, v in
                                    pd.Series(mm.index.normalize()).value_counts().head(10).items()},
        }
        if len(mm) == 0:
            res["verdict_hint"] = "plain mean reproduces the original everywhere"
        elif outside.sum() > 0:
            res["verdict_hint"] = ("SOME original hourly values lie outside their own raw hour's [min,max] -> "
                                   "built from different/earlier raw data (like WIND_DIR)")
        elif res["pct_mismatch"] < 0.5:
            res["verdict_hint"] = "tiny mismatch (<0.5% of hours), all inside raw range; likely edge/rounding effects"
        else:
            res["verdict_hint"] = "material mismatch but inside raw range; see G and the aggregator fractions"
        out[col] = res
    return out, mismatch_hours, means


def section_G(orig, clean_hourly_mean, mismatch_hours, raw_means, tol_rel, tol_abs):
    out = {}
    for col, hrs in mismatch_hours.items():
        if len(hrs) == 0:
            out[col] = {"n_mismatch_hours": 0, "verdict_hint": "nothing to explain"}
            continue
        sh = best_shift(raw_means[col], orig[col], tol_rel, tol_abs)
        c = clean_hourly_mean[col].shift(sh).reindex(hrs)
        o = orig[col].reindex(hrs)
        has = c.notna()
        ok = match(o[has], c[has], tol_rel, tol_abs)
        n_ok = int(ok.sum())
        res = {"n_mismatch_hours": int(len(hrs)), "n_with_cleaned_value": int(has.sum()),
               "n_original_matches_cleaned_mean": n_ok,
               "n_neither_raw_nor_cleaned": int(has.sum() - n_ok),
               "n_cleaned_hour_all_nan": int((~has).sum())}
        if has.sum() == 0:
            res["frac_explained_by_cleaning"] = None
            res["verdict_hint"] = ("cannot test: every mismatching hour is entirely NaN in the cleaned file "
                                   "(cleaning removed that data), so there is no cleaned mean to compare with")
        else:
            frac = n_ok / has.sum()
            res["frac_explained_by_cleaning"] = float(frac)
            res["verdict_hint"] = ("mismatch is explained by an adjustment equivalent to this cleaning"
                                   if frac >= 0.95 else
                                   "partly explained" if frac >= 0.5 else "NOT explained by this cleaning")
        out[col] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("orig_hourly")
    ap.add_argument("--clean", default=None, help="cleaned 1-second file (enables section G)")
    ap.add_argument("--key", default="DATA")
    ap.add_argument("--orig-key", default="DATA")
    ap.add_argument("--tol-rel", type=float, default=1e-6)
    ap.add_argument("--tol-abs", type=float, default=1e-6)
    args = ap.parse_args()
    args.raw = str(resolve_input(args.raw))
    args.orig_hourly = str(resolve_input(args.orig_hourly))
    if args.clean:
        args.clean = str(resolve_input(args.clean))

    orig = pd.read_hdf(args.orig_hourly, key=args.orig_key)
    print("Loading raw columns and sorting chronologically ...")
    raw = sorted_by_index(load_cols(args.raw, args.key, COLS))
    try:
        rep_F, mm_hours, raw_means = section_F(raw, orig, args.tol_rel, args.tol_abs)
        print_report("followup2_F_mismatch_hours", rep_F)
    except Exception as exc:
        print_report("followup2_F_mismatch_hours", {"error": repr(exc), "traceback": traceback.format_exc()})
        return
    if args.clean:
        try:
            clean = load_cols(args.clean, args.key, COLS)
            clean_hourly = hourly(clean, "mean")
            print_report("followup2_G_explained_by_cleaning",
                         section_G(orig, clean_hourly, mm_hours, raw_means, args.tol_rel, args.tol_abs))
        except Exception as exc:
            print_report("followup2_G_explained_by_cleaning", {"error": repr(exc), "traceback": traceback.format_exc()})
    else:
        print("(section G skipped: pass --clean <cleaned 1-second file> to run it)")


if __name__ == "__main__":
    main()
