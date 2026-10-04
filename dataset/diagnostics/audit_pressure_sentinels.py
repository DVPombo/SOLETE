"""
audit_pressure_sentinels.py -- READ-ONLY completeness audit of the Pressure
sentinel detector, run against the CLEANED 1-second file.

Changes nothing: it reads only `Pressure[mbar]` and `Pressure[mbar]_qc` from
the cleaned file and prints report blocks. Nothing is written, and
clean_solete_1sec.py's behaviour is not touched. Whether to change the
detector based on this output is the maintainer's call.

Usage:
    python audit_pressure_sentinels.py SOLETE_Pombo_1sec_clean.h5 --key DATA
    # Jupyter:  !python3 audit_pressure_sentinels.py SOLETE_Pombo_1sec_clean.h5 > audit.txt 2>&1

Memory: two columns of the ~39M-row file (~400 MB), not the whole frame.

Three report blocks (paste all back, BEGIN/END lines included):

  pressure_audit_1_survivor_distribution
      Min/max/std (overall and per day) of the Pressure values that survived
      cleaning, the longest remaining constant run (+ the 10 longest, with
      timestamp/value), and a run-length histogram. READ THE HISTOGRAM BEFORE
      JUDGING THE 60-299s SCAN: if the sensor reports at ~0.1 mbar resolution,
      short identical runs happen naturally at 1 Hz, so what matters is
      whether the tail looks like a smooth decay (genuine) or has a bump /
      one repeated value (a residual plateau).

  pressure_audit_2_flat_runs_60_299s
      Every remaining run of 60-299 identical samples: count, rows, the
      values, per-day counts, and whether the run is adjacent to a removed
      (NaN) stretch -- a plateau touching a removed plateau is the strongest
      sign of a missed one. Also what lowering the threshold to 60 would flag.

  pressure_audit_3_possible_false_positives
      Short removed runs (QC_PLACEHOLDER, NaN) that sit BETWEEN two surviving
      readings. A genuine reading of exactly 1000.0 mbar is NOT a remote
      possibility on a day when pressure sweeps ~991-1005 mbar at 0.1 mbar
      resolution -- the exact-multiple-of-1000 rule would NaN each one. This
      block lists such gaps and how close their neighbours are to 1000.
"""
import argparse

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "pipeline"))
from qc_flags import QC_PLACEHOLDER
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root: gives `import solete.paths`
from solete.paths import resolve_input
from solete_report import print_report

PRES = "Pressure[mbar]"
PRES_QC = PRES + "_qc"

HIST_BINS = [(1, 1), (2, 4), (5, 9), (10, 29), (30, 59), (60, 299), (300, None)]


def finite_constant_runs(v):
    """(starts, lengths) of maximal runs of consecutive bit-identical FINITE values."""
    n = len(v)
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    finite = np.isfinite(v)
    new_run = np.ones(n, dtype=bool)
    new_run[1:] = ~((v[1:] == v[:-1]) & finite[1:] & finite[:-1])
    starts = np.flatnonzero(new_run)
    lengths = np.diff(np.append(starts, n))
    keep = finite[starts]
    return starts[keep], lengths[keep]


def true_runs(mask):
    m = np.asarray(mask, dtype=np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1  # inclusive ends


def _hist(lengths):
    out = {}
    for lo, hi in HIST_BINS:
        label = f"{lo}" if hi == lo else (f"{lo}-{hi}" if hi else f"{lo}+")
        sel = (lengths >= lo) if hi is None else ((lengths >= lo) & (lengths <= hi))
        out[label] = {"n_runs": int(sel.sum()), "n_rows": int(lengths[sel].sum())}
    return out


def block_1(v, idx, starts, lengths):
    finite = np.isfinite(v)
    vals = v[finite]
    rep = {"n_rows": int(len(v)), "n_finite": int(finite.sum()), "n_nan": int((~finite).sum())}
    if finite.sum() == 0:
        rep["note"] = "no finite pressure values survive"
        return rep
    rep["overall"] = {"min": float(vals.min()), "max": float(vals.max()),
                      "mean": float(vals.mean()), "std": float(vals.std())}
    uniq = np.unique(vals)
    d = np.diff(uniq)
    rep["n_distinct_values"] = int(len(uniq))
    rep["smallest_gap_between_distinct_values_mbar"] = float(d.min()) if len(d) else None
    rep["longest_constant_run_s"] = int(lengths.max()) if len(lengths) else 0
    order = np.argsort(-lengths, kind="stable")[:10]
    rep["ten_longest_constant_runs"] = [
        {"start": str(idx[starts[i]]), "value": float(v[starts[i]]), "length_s": int(lengths[i])} for i in order]
    rep["run_length_histogram"] = _hist(lengths)
    days = idx.normalize()
    per_day = {}
    df = pd.DataFrame({"v": v, "day": days})[finite]
    run_day = pd.Series(idx[starts].normalize())
    for day, g in df.groupby("day"):
        x = g["v"].to_numpy()
        ld = lengths[(run_day == day).to_numpy()]
        per_day[str(day.date())] = {"n_valid_s": int(len(x)), "min": float(x.min()), "max": float(x.max()),
                                    "std": float(x.std()), "longest_constant_run_s": int(ld.max()) if len(ld) else 0}
    rep["n_days_with_survivors"] = len(per_day)
    rep["per_day"] = per_day
    return rep


def block_2(v, idx, starts, lengths, lo=60, hi=299, max_listed=100):
    n = len(v)
    sel = (lengths >= lo) & (lengths <= hi)
    s, ln = starts[sel], lengths[sel]
    rep = {"window_s": [lo, hi], "n_runs": int(sel.sum()), "n_rows": int(ln.sum()),
           "n_finite_rows_total": int(np.isfinite(v).sum())}
    if sel.sum() == 0:
        rep["verdict_hint"] = "no flat run of 60-299 s remains; lowering the threshold to 60 would catch nothing new"
        return rep
    rep["pct_of_finite_rows"] = float(100 * ln.sum() / max(1, np.isfinite(v).sum()))
    ends = s + ln - 1
    left_nan = (s > 0) & ~np.isfinite(v[np.maximum(s - 1, 0)])
    right_nan = (ends < n - 1) & ~np.isfinite(v[np.minimum(ends + 1, n - 1)])
    rep["n_runs_touching_removed_stretch"] = int((left_nan | right_nan).sum())
    vals = v[s]
    uv, cnt = np.unique(vals, return_counts=True)
    top = np.argsort(-cnt)[:15]
    rep["values_by_run_count"] = {float(uv[i]): int(cnt[i]) for i in top}
    rep["n_distinct_values"] = int(len(uv))
    rep["runs_per_day"] = {str(k.date()): int(c) for k, c in pd.Series(idx[s].normalize()).value_counts().sort_index().items()}
    order = np.argsort(-ln, kind="stable")[:max_listed]
    rep["runs_longest_first"] = [
        {"start": str(idx[s[i]]), "value": float(vals[i]), "length_s": int(ln[i]),
         "touches_removed_stretch": bool(left_nan[i] or right_nan[i])} for i in order]
    rep["listing_truncated"] = bool(sel.sum() > max_listed)
    return rep


def block_3(v, idx, qc, max_run=60, near=2.0, max_listed=50):
    """Short placeholder-removed gaps with a surviving reading on both sides."""
    n = len(v)
    removed = qc == QC_PLACEHOLDER
    rep = {"max_gap_length_s": max_run, "n_placeholder_rows_total": int(removed.sum())}
    if removed.sum() == 0:
        rep["note"] = "no QC_PLACEHOLDER rows in the file"
        return rep
    s, e = true_runs(removed)
    ln = e - s + 1
    ok = (s > 0) & (e < n - 1)
    ok &= np.isfinite(v[np.maximum(s - 1, 0)]) & np.isfinite(v[np.minimum(e + 1, n - 1)])
    short = ok & (ln <= max_run)
    rep["n_placeholder_runs_total"] = int(len(s))
    rep["n_short_gaps_between_surviving_readings"] = int(short.sum())
    if short.sum() == 0:
        rep["verdict_hint"] = "no short removed gap sits between two surviving readings"
        return rep
    s, e, ln = s[short], e[short], ln[short]
    lv, rv = v[s - 1], v[e + 1]
    around_1000 = (np.abs(lv - 1000.0) <= near) & (np.abs(rv - 1000.0) <= near)
    rep["n_gaps_with_both_neighbours_within_tolerance_of_1000"] = int(around_1000.sum())
    rep["tolerance_mbar"] = near
    rep["n_rows_in_these_gaps"] = int(ln.sum())
    rep["n_rows_in_gaps_near_1000"] = int(ln[around_1000].sum())
    rep["gap_length_histogram_s"] = {int(k): int(c) for k, c in pd.Series(ln).value_counts().sort_index().head(30).items()}
    rep["gaps_per_day"] = {str(k.date()): int(c) for k, c in pd.Series(idx[s].normalize()).value_counts().sort_index().items()}
    order = np.argsort(-ln, kind="stable")[:max_listed]
    rep["gaps_longest_first"] = [
        {"start": str(idx[s[i]]), "length_s": int(ln[i]), "left": float(lv[i]), "right": float(rv[i])} for i in order]
    rep["listing_truncated"] = bool(len(s) > max_listed)
    rep["interpretation_hint"] = (
        "Gaps whose neighbours straddle 1000.0 (e.g. 999.9 | 1000.1) are very likely genuine readings of exactly "
        "1000.0 that the multiple-of-1000 rule removed. Gaps far from 1000 are not explained by that mechanism.")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--key", default="DATA")
    ap.add_argument("--fp-max-gap", type=int, default=60)
    ap.add_argument("--fp-neighbour-tol", type=float, default=2.0)
    args = ap.parse_args()
    args.path = str(resolve_input(args.path))

    print(f"Reading {PRES} and {PRES_QC} from {args.path} (read-only) ...")
    df = pd.read_hdf(args.path, key=args.key, columns=[PRES, PRES_QC])
    idx = pd.DatetimeIndex(df.index)
    v = df[PRES].to_numpy(dtype=np.float64)
    qc = df[PRES_QC].to_numpy()
    meta = {"file": args.path, "index_monotonic_increasing": bool(idx.is_monotonic_increasing),
            "qc_code_counts": {int(k): int(c) for k, c in pd.Series(qc).value_counts().sort_index().items()}}
    print_report("pressure_audit_0_meta", meta)

    if not idx.is_monotonic_increasing:
        print("WARNING: index not chronological -- run detection below would be meaningless; aborting.")
        return

    starts, lengths = finite_constant_runs(v)
    print_report("pressure_audit_1_survivor_distribution", block_1(v, idx, starts, lengths))
    print_report("pressure_audit_2_flat_runs_60_299s", block_2(v, idx, starts, lengths))
    print_report("pressure_audit_3_possible_false_positives",
                 block_3(v, idx, qc, args.fp_max_gap, args.fp_neighbour_tol))


if __name__ == "__main__":
    main()
