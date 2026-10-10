# -*- coding: utf-8 -*-
"""
release_verify.py -- the verification of the v4 release (build_release.py stage `verify`).

Every check ends as one row {check, scope, status, detail}; status is PASS, FAIL, SKIP (could not run, with
the reason) or INFO (a number to read, not a test). verify() prints the table and raises if any row FAILs.

What can only be judged on the real 1 s file is printed, not asserted: see `cleaning_counts` (compared with
dataset/docs/CLEANING_DECISIONS.md by the maintainer) and the INFO rows of `model_vs_resampled_effect`.

The reproducibility check has two strengths (--rebuild-check): `sample` re-runs the cleaning, the fused
clean-then-expand and the resampling on a few windows of _original and compares them with the shipped rows
(cheap, always catches a rule or slicing error that shows up in those windows); `full` rebuilds every file in a
scratch folder with a DIFFERENT slice size and compares everything (doubles the build time).
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from solete.expansion import PHYSICAL_COLUMNS, expand_physical  # noqa: E402
from solete.h5io import h5_info, read_index, read_rows  # noqa: E402
from solete.params import import_PV_WT_data  # noqa: E402
import clean_solete_1sec as cl  # noqa: E402
import export_parquet as ep  # noqa: E402
import make_original as mo  # noqa: E402
import release_stages as st  # noqa: E402
import resample_solete as rs  # noqa: E402
from release_meta import ANGLE_COLUMNS  # noqa: E402

KEY = "DATA"
# The real record starts 2018-06-01 00:00:00 and covers 457 full days; the raw v3 file may or may not also hold the
# boundary second 2019-09-01 00:00:00 (the v3 hourly file does: 10,969 rows). Both are accepted; row counts are
# DERIVED from the span (expected_rows), never hard-coded.
REAL_START = pd.Timestamp("2018-06-01 00:00:00")
REAL_ENDS = (pd.Timestamp("2019-08-31 23:59:59"), pd.Timestamp("2019-09-01 00:00:00"))
STEP = {"1sec": pd.Timedelta("1s"), "1min": pd.Timedelta("1min"), "5min": pd.Timedelta("5min"), "60min": pd.Timedelta("60min")}
MODEL_FLOAT = ["Pac", "Pdc", "TempModule", "TempCell", "P_Solar_clean[kW]", "P_hybrid[kW]"]
# documented in dataset/docs/CLEANING_DECISIONS.md (real file only). The wind-direction figure documented there (1,153)
# counts values ABOVE 360 degrees only; code 1 also covers values below 0 and exactly 360, so it is compared as an INFO
# and the code-1 count is asserted against the counts taken from `_original` instead (cleaning_counts).
DOCUMENTED = {"pressure_days_with_real_value": ["2019-01-16"], "p_gaia_active_days": ["2018-08-31", "2019-05-25"]}
DOCUMENTED_WIND_DIR_ABOVE_360 = 1153


class Results:
    def __init__(self):
        self.rows = []

    def add(self, check, scope, status, detail=""):
        self.rows.append({"check": check, "scope": scope, "status": status, "detail": str(detail)})

    def run(self, check, scope, fn, *args, **kw):
        """Run fn; PASS with its returned detail, FAIL on AssertionError/any exception."""
        try:
            detail = fn(*args, **kw)
            self.add(check, scope, "PASS", "" if detail is None else detail)
            return detail
        except AssertionError as e:
            self.add(check, scope, "FAIL", e)
        except Exception as e:  # a crash in a check is a failure of the build, not something to skip
            self.add(check, scope, "FAIL", f"{type(e).__name__}: {e}")
        return None

    @property
    def failed(self):
        return [r for r in self.rows if r["status"] == "FAIL"]

    def print_table(self):
        w1 = max([len(r["check"]) for r in self.rows] + [5])
        w2 = max([len(r["scope"]) for r in self.rows] + [5])
        print(f"\n{'check':<{w1}}  {'scope':<{w2}}  status  detail")
        for r in self.rows:
            print(f"{r['check']:<{w1}}  {r['scope']:<{w2}}  {r['status']:<6}  {r['detail'][:200]}")
        n = {s: sum(r["status"] == s for r in self.rows) for s in ("PASS", "FAIL", "SKIP", "INFO")}
        print(f"\n{n['PASS']} passed, {n['FAIL']} FAILED, {n['SKIP']} skipped, {n['INFO']} informational\n", flush=True)


# ---------------------------------------------------------------------------------------------
# comparison helpers
# ---------------------------------------------------------------------------------------------
def diff_frames(a, b, label=""):
    """Problems found comparing two frames (columns and order, dtypes, index, values with equal NaN); [] if identical."""
    p = []
    if list(a.columns) != list(b.columns):
        p.append(f"{label}columns differ: only in first {sorted(set(a.columns) - set(b.columns))}, "
                 f"only in second {sorted(set(b.columns) - set(a.columns))} (or order)")
        return p
    if not a.index.equals(b.index):
        p.append(f"{label}index differs")
        return p
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype != y.dtype:
            p.append(f"{label}{c}: dtype {x.dtype} vs {y.dtype}")
        elif x.dtype.kind == "f":
            if not (((x == y) | (np.isnan(x) & np.isnan(y))).all()):
                p.append(f"{label}{c}: {int((~((x == y) | (np.isnan(x) & np.isnan(y)))).sum())} values differ")
        elif not (x == y).all():
            p.append(f"{label}{c}: {int((x != y).sum())} values differ")
    return p


def compare_h5(a_path, b_path, chunk_rows=2_000_000):
    """Exact comparison of two h5 files, chunk by chunk. Returns a short detail string; raises on a difference."""
    ia, ib = h5_info(a_path, KEY), h5_info(b_path, KEY)
    assert ia["columns"] == ib["columns"], f"columns differ: {Path(a_path).name} vs {Path(b_path).name}"
    assert ia["nrows"] == ib["nrows"], f"row counts differ: {ia['nrows']} vs {ib['nrows']}"
    for s in range(0, ia["nrows"], chunk_rows):
        a = read_rows(a_path, KEY, s, min(s + chunk_rows, ia["nrows"]))
        b = read_rows(b_path, KEY, s, min(s + chunk_rows, ia["nrows"]))
        problems = diff_frames(a, b, f"rows {s:,}+: ")
        assert not problems, "; ".join(problems[:5])
    return f"{ia['nrows']:,} rows x {len(ia['columns'])} columns identical"


# ---------------------------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------------------------
def is_real_span(first, last):
    return first == REAL_START and last in REAL_ENDS


def expected_rows(res, first, last):
    """Rows of a gap-free file of resolution `res` that covers the 1 s span [first, last] (whole-file resample grid)."""
    return int((last.floor(STEP[res]) - first.floor(STEP[res])) / STEP[res]) + 1


def check_grid(path, res, like=None):
    """Grid check of one file. `like` = (first, last) of the 1 s file the coarse file must cover (default: itself)."""
    idx = read_index(path, KEY)
    assert idx.is_monotonic_increasing and idx.is_unique, "index not strictly increasing"
    steps = np.diff(idx.values)
    assert (steps == STEP[res].to_timedelta64()).all(), f"{int((steps != STEP[res].to_timedelta64()).sum())} gaps/irregular steps on the {res} grid"
    first, last = like if like else (idx[0], idx[-1])
    want = expected_rows(res, first, last)
    assert len(idx) == want, f"{len(idx):,} rows but the span {first} .. {last} needs {want:,} at {res}"
    detail = f"{len(idx):,} rows, {idx[0]} .. {idx[-1]}, unique, monotonic, gap-free"
    if is_real_span(first, last):
        detail += ("; real record span" + (" incl. the boundary second 2019-09-01 00:00:00" if last == REAL_ENDS[1] else ", no boundary second"))
    else:
        detail += "; span is not the real one"
    return detail


def expected_columns(res, original_columns):
    meas = list(original_columns)
    qc = [f"{c}_qc" for c in cl.QC_COLS]
    if res == "1sec":
        return meas + list(ANGLE_COLUMNS) + qc + list(PHYSICAL_COLUMNS)
    return (meas + list(ANGLE_COLUMNS) + [f"{c}_worst" for c in qc] + [f"{c}_frac_flagged" for c in qc]
            + list(PHYSICAL_COLUMNS))


def check_columns(path, res, original_columns):
    cols = h5_info(path, KEY)["columns"]
    exp = expected_columns(res, original_columns)
    assert cols == exp, (f"columns differ from the documented set: missing {[c for c in exp if c not in cols]}, "
                         f"extra {[c for c in cols if c not in exp]}")
    assert not any(c.startswith(("Azimuth[deg]_qc", "Elevation[deg]_qc")) for c in cols), "Azimuth/Elevation flag columns present (D1)"
    return f"{len(cols)} columns"


def check_dtypes_1sec(path):
    one = read_rows(path, KEY, 0, 1000)
    bad = []
    for c in one.columns:
        want = "int8" if (c.endswith("_qc") or c.endswith("_qc_source")) else "float64"
        if str(one[c].dtype) != want:
            bad.append(f"{c}: {one[c].dtype} (expected {want})")
    assert not bad, "; ".join(bad)
    return "floats float64, flags int8"


def check_original_structure(original, original_columns):
    info = h5_info(original, KEY)
    assert info["format"] == "table", f"_original is {info['format']}, expected table"
    assert list(info["columns"]) == list(original_columns) and len(info["columns"]) == 9, f"columns: {info['columns']}"
    assert set(info["columns"]) == set(mo.MEASURED), "not the nine measured columns"
    return "9 measured columns, table format"


def check_idempotent(path, res, chunk_rows=2_000_000):
    PV, _ = import_PV_WT_data()
    n = h5_info(path, KEY)["nrows"]
    for s in range(0, n, chunk_rows):
        stored = read_rows(path, KEY, s, min(s + chunk_rows, n))
        again = expand_physical(stored.copy(), PV)
        problems = diff_frames(stored[list(PHYSICAL_COLUMNS)], again[list(PHYSICAL_COLUMNS)], f"rows {s:,}+: ")
        assert not problems, "; ".join(problems[:5])
    return "expand_physical reproduces the stored model columns exactly"


def check_psolar_unchanged(final_1s, original, chunk_rows=2_000_000):
    n = h5_info(original, KEY)["nrows"]
    for s in range(0, n, chunk_rows):
        a = read_rows(original, KEY, s, min(s + chunk_rows, n), columns=["P_Solar[kW]"])["P_Solar[kW]"].to_numpy()
        b = read_rows(final_1s, KEY, s, min(s + chunk_rows, n), columns=["P_Solar[kW]"])["P_Solar[kW]"].to_numpy()
        assert ((a == b) | (np.isnan(a) & np.isnan(b))).all(), f"P_Solar[kW] differs from the measurement near row {s:,}"
    return "measured P_Solar[kW] equals _original exactly"


def check_resampled_scratch(res, final_path, scratch_path):
    """Measured columns and pipeline flags of the final coarse file equal the scratch (pre-expansion) resample; no code 6."""
    sc = read_rows(scratch_path, KEY)
    fin = read_rows(final_path, KEY, columns=list(sc.columns))
    problems = diff_frames(sc, fin[list(sc.columns)])
    assert not problems, "; ".join(problems[:5])
    assert not any(c in sc.columns for c in rs.SKIP_MODEL_COLUMNS), "model columns present before expansion"
    worst = [c for c in sc.columns if c.endswith("_qc_worst")]
    assert not (sc[worst] == 6).any().any(), "code 6 in a pre-expansion resampled file"
    return f"{len(sc.columns)} resampled columns identical in the final file; no code 6 before expansion"


def model_vs_resampled_effect(final_1s, final_coarse, res, slice_days=31):
    """How far each model column of the coarse file is from resample(expand(1 s)): the documented, expected effect."""
    idx = read_index(final_1s, KEY)
    slices, _ = rs.day_slices(idx, slice_days)
    rule = STEP[res]
    pieces = []
    for a, b in slices:
        blk = read_rows(final_1s, KEY, a, b, columns=MODEL_FLOAT)
        pieces.append(blk.resample(rule, label="left", closed="left").mean())
    avg = pd.concat(pieces)
    coarse = read_rows(final_coarse, KEY, columns=MODEL_FLOAT)
    avg = avg.reindex(coarse.index)
    out = {}
    for c in MODEL_FLOAT:
        d = (coarse[c] - avg[c]).to_numpy()
        ok = np.isfinite(d)
        base = np.abs(avg[c].to_numpy()[ok])
        out[c] = {"mean_diff": float(d[ok].mean()) if ok.any() else None,
                  "mean_abs_diff": float(np.abs(d[ok]).mean()) if ok.any() else None,
                  "rel_mean_abs_diff_pct": float(100 * np.abs(d[ok]).mean() / base.mean()) if ok.any() and base.mean() > 0 else None,
                  "max_abs_diff": float(np.abs(d[ok]).max()) if ok.any() else None, "n_buckets": int(ok.sum())}
    return out


def wind_dir_outside_counts(original, chunk_rows=2_000_000):
    """From the raw measurements: how many WIND_DIR values are above 360, below 0, or exactly 360."""
    n = h5_info(original, KEY)["nrows"]
    gt = lt = eq = 0
    for s in range(0, n, chunk_rows):
        v = read_rows(original, KEY, s, min(s + chunk_rows, n), columns=["WIND_DIR[deg]"])["WIND_DIR[deg]"].to_numpy()
        gt += int((v > 360).sum())
        lt += int((v < 0).sum())
        eq += int((v == 360).sum())
    return {"above_360": gt, "below_0": lt, "equal_360": eq}


def cleaning_counts(final_1s, chunk_rows=2_000_000):
    """Counts that CLEANING_DECISIONS.md documents for the REAL file, computed from the shipped 1 s file."""
    n = h5_info(final_1s, KEY)["nrows"]
    wrapped, real_days, active_days = 0, set(), set()
    for s in range(0, n, chunk_rows):
        b = read_rows(final_1s, KEY, s, min(s + chunk_rows, n),
                      columns=["WIND_DIR[deg]_qc", "Pressure[mbar]", "Pressure[mbar]_qc", "P_Gaia[kW]_qc"])
        wrapped += int((b["WIND_DIR[deg]_qc"] == 1).sum())
        real = b["Pressure[mbar]"].notna() & (b["Pressure[mbar]_qc"] == 0)
        real_days |= {str(d.date()) for d in b.index[real.to_numpy()].normalize().unique()}
        act = (b["P_Gaia[kW]_qc"] == 10).to_numpy()
        active_days |= {str(d.date()) for d in b.index[act].normalize().unique()}
    return {"wind_dir_rows_wrapped": wrapped, "pressure_days_with_real_value": sorted(real_days),
            "p_gaia_active_days": sorted(active_days)}


def platform_check(res="60min"):
    """import_SOLETE_data(data_version='v4') loads the built file and its working P_Solar[kW] equals P_Solar_clean[kW]."""
    try:
        from solete.io import import_SOLETE_data
    except ImportError as e:
        return None, f"platform dependencies not installed ({e})"
    PV, WT = import_PV_WT_data()
    cv = {"resolution": res, "data_version": "v4", "SOLETE_builvsimport": "Build", "SOLETE_save": False,
          "OriginalFeatures": [], "PossibleFeatures": []}
    out = import_SOLETE_data(cv, PV, WT)
    a, b = out["P_Solar[kW]"].to_numpy(), out["P_Solar_clean[kW]"].to_numpy()
    assert ((a == b) | (np.isnan(a) & np.isnan(b))).all(), "working P_Solar[kW] != P_Solar_clean[kW]"
    return True, f"loaded {len(out):,} rows; working P_Solar[kW] == P_Solar_clean[kW]"


# ---------------------------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------------------------
def sample_rebuild(original, final_1s, final_coarse, window_days=2, targets=None):
    """Re-run clean -> (fused) expand and clean -> resample -> expand on a few windows of _original, cut at safe
    boundaries, and compare with the shipped rows. Returns a detail string; raises on any difference."""
    PV, _ = import_PV_WT_data()
    idx = read_index(original, KEY)
    n = len(idx)
    if targets is None:
        day = pd.Timestamp("2019-01-16")
        mid = int(idx.searchsorted(day)) if idx[0] <= day <= idx[-1] else n // 2
        targets = sorted({0, mid, max(n - window_days * st.DAY, 0)})
    done, rows = [], 0
    for t in targets:
        start = 0 if t == 0 else cl.find_safe_cut(original, KEY, t, n)
        if start >= n:
            continue
        end = cl.find_safe_cut(original, KEY, min(start + window_days * st.DAY, n - 1), n) if start + window_days * st.DAY < n else n
        if end <= start:
            continue
        block = read_rows(original, KEY, start, end)
        cleaned, _ = cl.clean_block(block, verbose=False)
        fused = expand_physical(cleaned.copy(), PV)                       # fused path: clean then expand, in memory
        shipped = read_rows(final_1s, KEY, start, end)
        problems = diff_frames(fused, shipped, f"1 s rows {start:,}..{end:,}: ")
        assert not problems, "; ".join(problems[:5])
        t0, t1 = cleaned.index[0], cleaned.index[-1] + pd.Timedelta("1s")
        for res, path in final_coarse.items():
            period = STEP[res]
            r = rs.resample_dataframe(cleaned, res)
            r = r[(r.index >= t0.ceil(period)) & (r.index + period <= t1)]
            if not len(r):
                continue
            expand_physical(r, PV)
            ship = read_rows(path, KEY).loc[r.index]
            problems = diff_frames(r, ship, f"{res} window {start:,}..{end:,}: ")
            assert not problems, "; ".join(problems[:5])
        done.append(f"{cleaned.index[0].date()}..{cleaned.index[-1].date()}")
        rows += end - start
    assert done, "no window could be rebuilt"
    return f"{len(done)} windows ({rows:,} rows) rebuilt via clean+expand (fused) and compared: {', '.join(done)}"


def full_rebuild(original, shipped, slice_days, workdir):
    """Rebuild everything from _original in `workdir` with `slice_days` (use a value different from the build's) and
    compare each file with the shipped one. `shipped` = {'1sec': path, '1min': path, ...}."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cleaned = workdir / "cleaned_1sec.h5"
    st.stage_clean(original, cleaned, slice_days, overwrite=True)           # fractional days are fine here
    scratch = {r: workdir / f"resampled_{r}.h5" for r in st.COARSE}
    st.stage_resample(cleaned, scratch, max(int(slice_days), 1), overwrite=True)   # resampling cuts on whole days
    out = {}
    one = workdir / "final_1sec.h5"
    st.stage_expand_1s(cleaned, one, slice_days, overwrite=True)
    out["1sec"] = compare_h5(one, shipped["1sec"])
    for r in st.COARSE:
        fin = workdir / f"final_{r}.h5"
        st.stage_expand_coarse(scratch[r], fin, overwrite=True)
        out[r] = compare_h5(fin, shipped[r])
    return out


# ---------------------------------------------------------------------------------------------
def verify(files, raw=None, scratch=None, rebuild_check="sample", state=None, slice_days=31, parquet=None):
    """Run every check. files: {'original': p, '1sec': p, '1min': p, '5min': p, '60min': p} (h5);
    scratch: {'cleaned': p, 'resampled': {rule: p}} or None; parquet: {name: p} or None.
    Returns (Results, extras dict). Does not raise: the caller decides (build_release raises after saving)."""
    R = Results()
    extras = {}
    orig_cols = h5_info(files["original"], KEY)["columns"]

    R.run("original: nine measured columns, table format", "original", check_original_structure, files["original"], orig_cols)
    R.run("original: grid (sorted, unique, gap-free)", "original", check_grid, files["original"], "1sec")
    rec = (state or {}).get("original_verification")
    if rec:
        R.add("original: identical to the raw file (values, dtypes, aligned by timestamp)", "original", "PASS",
              f"verified when it was built: {rec.get('n_rows_compared', 0):,} rows")
    elif raw and Path(raw).exists():
        R.run("original: identical to the raw file (values, dtypes, aligned by timestamp)", "original",
              lambda: f"{mo.verify_original(raw, files['original'], KEY, verbose=False)['n_rows_compared']:,} rows identical")
    else:
        R.add("original: identical to the raw file", "original", "SKIP", "raw file not available and no build record")

    i1 = read_index(files["1sec"], KEY)
    span1 = (i1[0], i1[-1])
    for res in ("1sec",) + st.COARSE:
        R.run("rows / span / grid", res, check_grid, files[res], res, span1)
        R.run("column set (no Azimuth/Elevation flag columns)", res, check_columns, files[res], res, orig_cols)
        R.run("expand_physical idempotent", res, check_idempotent, files[res], res)
    R.run("dtypes", "1sec", check_dtypes_1sec, files["1sec"])
    R.run("measured P_Solar[kW] untouched", "1sec", check_psolar_unchanged, files["1sec"], files["original"])

    if scratch and Path(scratch["cleaned"]).exists():
        R.run("P_Solar[kW] equals the cleaned measurement", "1sec",
              lambda: (compare_h5_cols(files["1sec"], scratch["cleaned"], ["P_Solar[kW]"])))
        for res in st.COARSE:
            sp = scratch["resampled"][res]
            if Path(sp).exists():
                R.run("measured+flag columns equal the resample of cleaned 1 s; no code 6 before expansion", res,
                      check_resampled_scratch, res, files[res], sp)
            else:
                R.add("measured+flag columns equal the resample of cleaned 1 s", res, "SKIP", "scratch resample not kept")
    else:
        R.add("measured+flag columns equal the resample of cleaned 1 s; no code 6 before expansion", "coarse", "SKIP",
              "scratch intermediates not kept (covered by --rebuild-check full)")

    if rebuild_check == "none":
        R.add("reproducibility: rebuild from _original", "all", "SKIP", "--rebuild-check none")
    elif rebuild_check == "sample":
        R.run("reproducibility: sample windows rebuilt from _original (fused path)", "all", sample_rebuild,
              files["original"], files["1sec"], {r: files[r] for r in st.COARSE})
    else:
        other = max(slice_days * 0.6, 0.05)          # a different slice size than the build's (also proves slice independence)
        with tempfile.TemporaryDirectory(dir=str(Path(files["1sec"]).parent), prefix="verify_rebuild_") as tmp:
            try:
                res_ = full_rebuild(files["original"], files, other, tmp)
                for k, v in res_.items():
                    R.add("reproducibility: full rebuild from _original (slice size differs from the build's)", k, "PASS", v)
            except AssertionError as e:
                R.add("reproducibility: full rebuild from _original", "all", "FAIL", e)

    for res in st.COARSE:
        eff = None
        try:
            eff = model_vs_resampled_effect(files["1sec"], files[res], res, slice_days)
        except Exception as e:
            R.add("model columns vs resample(expand(1 s))", res, "FAIL", f"{type(e).__name__}: {e}")
        if eff is not None:
            extras.setdefault("model_vs_resampled", {})[res] = eff
            R.add("model columns vs resample(expand(1 s)) (documented, expected)", res, "INFO",
                  "; ".join(f"{c}: mean|d|={v['mean_abs_diff']:.4g}" for c, v in eff.items() if v["mean_abs_diff"] is not None))

    ok, msg = None, ""
    try:
        ok, msg = platform_check("60min")
    except AssertionError as e:
        R.add("platform: import_SOLETE_data(data_version='v4')", "60min", "FAIL", e)
    except Exception as e:
        R.add("platform: import_SOLETE_data(data_version='v4')", "60min", "FAIL", f"{type(e).__name__}: {e}")
    else:
        R.add("platform: import_SOLETE_data(data_version='v4')", "60min", "PASS" if ok else "SKIP", msg)

    counts = cleaning_counts(files["1sec"])
    extras["cleaning_counts"] = counts
    real = is_real_span(*span1)
    wd = None
    try:
        wd = wind_dir_outside_counts(files["original"])
    except Exception as e:
        R.add("wind direction: code-1 rows vs the raw values outside [0, 360)", "1sec", "FAIL", f"{type(e).__name__}: {e}")
    if wd is not None:
        extras["wind_dir_outside_counts_original"] = wd
        outside = sum(wd.values())
        R.add("wind direction: code-1 rows == raw values above 360 + below 0 + equal 360", "1sec",
              "PASS" if counts["wind_dir_rows_wrapped"] == outside else "FAIL",
              f"code 1: {counts['wind_dir_rows_wrapped']}; raw: above 360 = {wd['above_360']}, below 0 = {wd['below_0']}, "
              f"equal 360 = {wd['equal_360']}")
        R.add("wind direction: raw values above 360 vs CLEANING_DECISIONS.md", "1sec", "INFO",
              f"{wd['above_360']} (documented {DOCUMENTED_WIND_DIR_ABOVE_360})")
    for k in ("pressure_days_with_real_value", "p_gaia_active_days"):
        v, doc = counts[k], DOCUMENTED[k]
        if real:
            R.add(f"cleaning count vs CLEANING_DECISIONS.md: {k}", "1sec", "PASS" if v == doc else "FAIL", f"{v} (documented {doc})")
        else:
            R.add(f"cleaning count: {k} (synthetic data: documented value not applicable)", "1sec", "INFO", f"{v}")

    if parquet:
        for name, pq_path in parquet.items():
            h5 = files["original"] if name == "original" else files[name]
            R.run("Parquet round trip against the h5", name, lambda h=h5, p=pq_path: ep.verify_roundtrip(h, p, KEY))
    else:
        R.add("Parquet round trip against the h5", "all", "SKIP", "no Parquet files in this run")
    return R, extras


def compare_h5_cols(a, b, cols, chunk_rows=2_000_000):
    n = h5_info(a, KEY)["nrows"]
    assert n == h5_info(b, KEY)["nrows"], "row counts differ"
    for s in range(0, n, chunk_rows):
        x = read_rows(a, KEY, s, min(s + chunk_rows, n), columns=cols)
        y = read_rows(b, KEY, s, min(s + chunk_rows, n), columns=cols)
        problems = diff_frames(x, y)
        assert not problems, "; ".join(problems[:3])
    return f"{cols} identical"
