# -*- coding: utf-8 -*-
"""
build_release.py -- build every file of the SOLETE v4 release (figshare upload) with one command.

    SOLETE_Pombo_1sec_original_v4   raw 1 s data, sorted, nine measured columns, nothing cleaned
    SOLETE_Pombo_1sec_v4            _original -> clean + flags + recomputed Azimuth/Elevation -> expand_physical
    SOLETE_Pombo_{1min,5min,60min}_v4   measured + pipeline flags resampled from the cleaned 1 s data,
                                    then expand_physical on each file from its own inputs
    ... each as HDF5 (data/hdf5/) and Parquet (data/parquet/), plus SHA256SUMS.txt, manifest.json,
    the generated resampling methodology and build_summary_v4.json next to the data.

Stages (--stages, default all; each heavy stage runs in its own subprocess so memory is returned):
    original  raw v3 1 s file -> SOLETE_Pombo_1sec_original_v4.h5   (sorted, Azimuth/Elevation dropped, verified)
    clean     _original -> cleaned 1 s, sliced, into a scratch folder
    resample  cleaned 1 s -> 1min/5min/60min measured + flag files (scratch)
    expand    expand_physical on each of the four files -> the four final .h5 files
    parquet   the five .h5 -> five .parquet (measured compression trial, round trip in `verify`)
    verify    the verification table (see release_verify.py); fails loudly
    manifest  SHA256SUMS.txt + manifest.json; removes the scratch folder unless --keep-intermediate

Nothing is overwritten without --overwrite; files are written as *.tmp and renamed on success; a stage whose
outputs exist is skipped with --skip-existing, so an interrupted build resumes at the stage level.
Inputs are never modified. The raw v3 file keeps its name (SOLETE_Pombo_1sec.h5) and is read from --raw.

WHERE THINGS GO: all paths come from solete/paths.py: <data>/hdf5, <data>/parquet, scratch in
<data>/derived/build_v4 (--scratch). <data> is the repo's data/ folder or $SOLETE_DATA_DIR.

SPYDER (Run > Configuration per file...):
    1. tick "Execute in an external system terminal" (the build starts subprocesses and prints a lot)
    2. tick "Command line options" and paste, e.g.:
           --raw D:/solete/SOLETE_Pombo_1sec.h5 --skip-existing
       (set the SOLETE_DATA_DIR environment variable first if the data live outside the repo)
    3. Run. First time, add --dry-run to see the plan and the disk estimate without writing anything.
    Equivalent terminal command:
           python dataset/pipeline/build_release.py --raw D:/solete/SOLETE_Pombo_1sec.h5 --skip-existing
    Smaller machine: --slice-days 7 (slices of a week; same output, a little slower).
    After a successful build, replace the synthetic effect table in dataset/docs/METHODOLOGY.md with
        python scripts/expansion_checks.py effect --input <a cleaned 1 s slice of the real build>
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: gives `import solete.paths`

from solete import paths  # noqa: E402
from solete_report import print_report  # noqa: E402

STAGES = ["original", "clean", "resample", "expand", "parquet", "verify", "manifest"]
COARSE = ("1min", "5min", "60min")
ALL_RES = ("1sec",) + COARSE
RESULT_TAG = "@@SOLETE_STAGE_RESULT@@"
DAY_ROWS = 86400
N_ROWS_REAL = {"1sec": 39_484_800, "1min": 658_080, "5min": 131_616, "60min": 10_968}


# ---------------------------------------------------------------------------------------------
# paths of everything the build reads and writes
# ---------------------------------------------------------------------------------------------
def release_h5():
    d = {"original": paths.release_path("1sec", original=True)}
    d.update({r: paths.release_path(r) for r in ALL_RES})
    return d


def release_parquet():
    d = {"original": paths.release_path("1sec", "parquet", original=True)}
    d.update({r: paths.release_path(r, "parquet") for r in ALL_RES})
    return d


def scratch_files(scratch):
    scratch = Path(scratch)
    return {"cleaned": scratch / "cleaned_1sec.h5",
            "resampled": {r: scratch / f"resampled_{r}.h5" for r in COARSE},
            "state": scratch / "build_state.json"}


def methodology_path():
    return paths.DATA_DIR / "SOLETE_Pombo_v4_RESAMPLING_METHODOLOGY.md"


def summary_path():
    return paths.DATA_DIR / "build_summary_v4.json"


def remove_stale_manifest(reason):
    """SHA256SUMS.txt / manifest.json describe a set of files; once those files change or a check fails they must go,
    so a stale checksum list can never be published next to different data."""
    removed = []
    for name in ("SHA256SUMS.txt", "manifest.json"):
        f = paths.DATA_DIR / name
        if f.exists():
            f.unlink()
            removed.append(name)
    if removed:
        print(f"removed stale {', '.join(removed)} ({reason}); the manifest stage writes new ones", flush=True)


def outputs_of(stage, scratch):
    h5, pq_, sc = release_h5(), release_parquet(), scratch_files(scratch)
    return {
        "original": [h5["original"]],
        "clean": [sc["cleaned"]],
        "resample": list(sc["resampled"].values()),
        "expand": [h5[r] for r in ALL_RES],
        "parquet": list(pq_.values()),
        "verify": [], "manifest": [],
    }[stage]


# ---------------------------------------------------------------------------------------------
# disk estimate (an UPPER bound: bytes uncompressed; zlib level 1 / zstd normally give less)
# ---------------------------------------------------------------------------------------------
def disk_estimate(n_rows):
    meas, angle, qc_cols, model_f, model_i = 9, 2, 8, 6, 3
    per_row = {
        "original (h5)": meas * 8,
        "scratch cleaned 1 s (h5)": (meas + angle) * 8 + qc_cols,
        "final 1 s (h5)": (meas + angle + model_f) * 8 + qc_cols + model_i,
        "parquet original": meas * 8 + 8,
        "parquet 1 s": (meas + angle + model_f) * 8 + qc_cols + model_i + 8,
    }
    est = {k: int(v * n_rows) for k, v in per_row.items()}
    coarse_rows = sum(N_ROWS_REAL[r] for r in COARSE) * (n_rows / N_ROWS_REAL["1sec"])
    est["coarse files, scratch + final + parquet (h5 + parquet)"] = int(coarse_rows * 40 * 8 * 3)
    return est


def check_disk(needed_by_dir, margin=1.10):
    """needed_by_dir: {Path: bytes}. Refuse (return list of problems) when a volume is short."""
    by_dev = {}
    for d, nbytes in needed_by_dir.items():
        d = Path(d)
        while not d.exists() and d != d.parent:
            d = d.parent
        dev = os.stat(d).st_dev
        by_dev.setdefault(dev, [d, 0])[1] += nbytes
    problems = []
    for d, nbytes in by_dev.values():
        free = shutil.disk_usage(d).free
        if free < nbytes * margin:
            problems.append(f"{d}: {free / 1e9:.1f} GB free, about {nbytes * margin / 1e9:.1f} GB needed (upper bound incl. 10 % margin)")
    return problems


# ---------------------------------------------------------------------------------------------
# peak memory
# ---------------------------------------------------------------------------------------------
def peak_rss_mb():
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / (1e6 if sys.platform == "darwin" else 1e3), 1)
    except ImportError:
        try:
            import psutil
            return round(psutil.Process().memory_info().peak_wset / 1e6, 1)       # Windows
        except Exception:
            return None


# ---------------------------------------------------------------------------------------------
# the worker: runs ONE stage in this (child) process
# ---------------------------------------------------------------------------------------------
def load_state(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def save_state(path, state):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(state, indent=1, default=str))


def run_stage(stage, a):
    import release_stages as st  # heavy imports only in the worker
    import release_verify as rv
    h5, pq_, sc = release_h5(), release_parquet(), scratch_files(a.scratch)
    Path(a.scratch).mkdir(parents=True, exist_ok=True)
    state = load_state(sc["state"])
    ow = a.overwrite
    out = {}
    if stage == "original":
        raw = Path(a.raw) if a.raw else paths.HDF5_DIR / paths.data_filename("1sec", "v3")
        if not raw.exists():
            raise FileNotFoundError(f"raw v3 file not found: {raw} (pass --raw)")
        out = st.stage_original(raw, h5["original"], overwrite=ow)
        state["original_verification"] = out["verification"]
        state["raw"] = str(raw)
    elif stage == "clean":
        out = st.stage_clean(h5["original"], sc["cleaned"], a.slice_days, overwrite=ow)
        state["clean_report"] = out
    elif stage == "resample":
        out = st.stage_resample(sc["cleaned"], sc["resampled"], a.slice_days, overwrite=ow)
    elif stage == "expand":
        out["1sec"] = st.stage_expand_1s(sc["cleaned"], h5["1sec"], a.slice_days, overwrite=ow)
        for r in COARSE:
            out[r] = st.stage_expand_coarse(sc["resampled"][r], h5[r], overwrite=ow)
        import resample_solete as rs
        from solete.h5io import h5_info
        rs.write_methodology_doc(str(methodology_path()),
                                 [c for c in h5_info(sc["cleaned"], "DATA")["columns"] if c not in rs.SKIP_MODEL_COLUMNS])
    elif stage == "parquet":
        out = st.stage_parquet({k: h5[k] for k in ["original"] + list(ALL_RES)},
                               {k: pq_[k] for k in ["original"] + list(ALL_RES)},
                               trial_source=h5["1sec"], overwrite=ow)
    elif stage == "verify":
        parquet = {k: v for k, v in pq_.items() if Path(v).exists()} or None
        scratch = sc if Path(sc["cleaned"]).exists() else None
        R, extras = rv.verify(h5, raw=state.get("raw") or a.raw, scratch=scratch, rebuild_check=a.rebuild_check,
                              state=state, slice_days=a.slice_days, parquet=parquet)
        R.print_table()
        out = {"rows": R.rows, "extras": extras, "n_failed": len(R.failed)}
    elif stage == "manifest":
        out = write_manifest(h5, pq_, a)
    save_state(sc["state"], state)
    return out


def sha256_of(path, block=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(h5, pq_, a):
    from solete.h5io import h5_info
    files = []
    listing = [(k, p, "hdf5") for k, p in h5.items()] + [(k, p, "parquet") for k, p in pq_.items()]
    # the figshare text gets the real file sizes (template: <repo>/data/figshare_README.txt, block between the markers)
    template = paths.REPO_ROOT / "data" / "figshare_README.txt"
    if template.exists():
        text = template.read_text(encoding="utf-8")
        if "[[SIZES-BEGIN]]" in text and "[[SIZES-END]]" in text:
            rows = [f"  {'file':<52s}{'rows':>13s}{'columns':>9s}{'size':>13s}"]
            for k, p in list(h5.items()) + list(pq_.items()):
                inf = h5_info(h5[k], "DATA")
                ncol = len(inf["columns"]) + (1 if str(p).endswith(".parquet") else 0)
                rows.append(f"  {Path(p).relative_to(paths.DATA_DIR).as_posix():<52s}{inf['nrows']:>13,d}{ncol:>9d}{Path(p).stat().st_size / 1e6:>10,.1f} MB")
            head, rest = text.split("[[SIZES-BEGIN]]", 1)
            _, tail = rest.split("[[SIZES-END]]", 1)
            target = paths.DATA_DIR / "figshare_README.txt"
            same = target.resolve() == template.resolve()          # keep the markers if the template is rewritten in place
            body = "\n".join(rows)
            target.write_text(head + (f"[[SIZES-BEGIN]]\n{body}\n[[SIZES-END]]" if same else body) + tail, encoding="utf-8")
    # documents that belong to the release; build_summary_v4.json is not listed (it holds timings and changes every run)
    extras = [p for p in (methodology_path(), paths.DATA_DIR / "figshare_README.txt") if Path(p).exists()]
    sums = []
    for k, p, fmt in listing:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"manifest: {p} is missing; run the earlier stages")
        info = h5_info(h5[k], "DATA")
        digest = sha256_of(p)
        rel = p.relative_to(paths.DATA_DIR).as_posix()
        sums.append(f"{digest}  {rel}")
        files.append({"file": rel, "format": fmt, "role": k, "rows": info["nrows"], "columns": len(info["columns"]) + (1 if fmt == "parquet" else 0),
                      "bytes": p.stat().st_size, "sha256": digest})
    for p in extras:
        digest = sha256_of(p)
        rel = Path(p).relative_to(paths.DATA_DIR).as_posix()
        sums.append(f"{digest}  {rel}")
        files.append({"file": rel, "format": "document", "role": "document", "bytes": Path(p).stat().st_size, "sha256": digest})
    (paths.DATA_DIR / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")
    import pandas, pyarrow, pvlib, numpy
    manifest = {"version": "v4", "built_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "python": platform.python_version(), "platform": platform.platform(),
                "packages": {"pandas": pandas.__version__, "numpy": numpy.__version__, "pyarrow": pyarrow.__version__,
                             "pvlib": pvlib.__version__},
                "slice_days": a.slice_days, "files": files}
    (paths.DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return {"files": len(files)}


# ---------------------------------------------------------------------------------------------
# the orchestrator
# ---------------------------------------------------------------------------------------------
def child_command(stage, a):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--stage-worker", stage, "--slice-days", str(a.slice_days),
           "--scratch", str(a.scratch), "--rebuild-check", a.rebuild_check]
    if a.raw:
        cmd += ["--raw", str(a.raw)]
    if a.overwrite:
        cmd.append("--overwrite")
    return cmd


def run_child(stage, a):
    """Run one stage in a subprocess; stream its output; return (returncode, result dict or None)."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", SOLETE_DATA_DIR=str(paths.DATA_DIR))
    proc = subprocess.Popen(child_command(stage, a), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)
    result = None
    for line in proc.stdout:
        if line.startswith(RESULT_TAG):
            result = json.loads(line[len(RESULT_TAG):])
        else:
            print(line, end="", flush=True)
    return proc.wait(), result


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="all", help="comma list of: " + ", ".join(STAGES) + " (default: all)")
    ap.add_argument("--raw", default=None, help="raw v3 1 s file (default: the v3 name in the data/hdf5 folder); never modified")
    ap.add_argument("--slice-days", type=float, default=31.0, help="approximate slice length in days (default 31; 7 for a small machine)")
    ap.add_argument("--scratch", default=str(paths.DERIVED_DIR / "build_v4"), help="scratch folder for the intermediate files")
    ap.add_argument("--keep-intermediate", action="store_true", help="do not delete the scratch folder at the end")
    ap.add_argument("--skip-existing", action="store_true", help="skip a stage whose outputs already exist (resume)")
    ap.add_argument("--overwrite", action="store_true", help="replace existing output files")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the disk estimate; write nothing")
    ap.add_argument("--rebuild-check", choices=["none", "sample", "full"], default="sample",
                    help="reproducibility check in `verify`: sample windows (default) or a full second build (doubles the time)")
    ap.add_argument("--stage-worker", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.stages == "all":
        a.stage_list = list(STAGES)
    else:
        a.stage_list = [s.strip() for s in a.stages.split(",") if s.strip()]
        bad = [s for s in a.stage_list if s not in STAGES]
        if bad:
            ap.error(f"unknown stage(s) {bad}; choose from {STAGES}")
        a.stage_list = [s for s in STAGES if s in a.stage_list]
    return a


def plan(a):
    """Decide per stage: 'run' or 'skip'; collect problems (refusals) before anything is written."""
    problems, actions = [], {}
    sc = scratch_files(a.scratch)
    for stage in a.stage_list:
        outs = outputs_of(stage, a.scratch)
        exist = [p for p in outs if Path(p).exists()]
        if stage in ("verify", "manifest"):
            actions[stage] = "run"
        elif exist and len(exist) == len(outs) and a.skip_existing:
            actions[stage] = "skip"
        elif exist and not a.overwrite:
            problems.append(f"stage {stage}: output exists and neither --skip-existing nor --overwrite was given: "
                            + ", ".join(Path(p).name for p in exist))
            actions[stage] = "refuse"
        else:
            actions[stage] = "run"
    # the scratch files are deleted after a finished build: with --skip-existing, clean/resample are not needed
    # again when everything they feed (the final .h5 files) is already there
    final_there = all(Path(p).exists() for p in outputs_of("expand", a.scratch))
    for stage in ("clean", "resample"):
        if (stage in actions and a.skip_existing and final_there and actions[stage] != "skip"
                and actions.get("expand", "skip") == "skip"):
            actions[stage] = "skip"
            problems = [p for p in problems if not p.startswith(f"stage {stage}:")]
    prereq = {"clean": ("original", [release_h5()["original"]]), "resample": ("clean", [sc["cleaned"]]),
              "expand": ("resample", [sc["cleaned"]] + list(sc["resampled"].values())),
              "parquet": ("expand", [release_h5()[r] for r in ALL_RES] + [release_h5()["original"]]),
              "verify": ("expand", [release_h5()[r] for r in ALL_RES] + [release_h5()["original"]]),
              "manifest": ("parquet", [release_h5()[r] for r in ALL_RES] + list(release_parquet().values()))}
    for stage in a.stage_list:
        if stage in prereq and actions[stage] == "run":
            made_by, needed = prereq[stage]
            missing = [p for p in needed if not Path(p).exists()]
            if missing and not (made_by in a.stage_list):
                problems.append(f"stage {stage} needs {', '.join(Path(p).name for p in missing)} (made by stage {made_by}, which is not selected)")
    if "original" in a.stage_list and actions.get("original") == "run":
        raw = Path(a.raw) if a.raw else paths.HDF5_DIR / paths.data_filename("1sec", "v3")
        if not raw.exists() and not a.dry_run:
            problems.append(f"raw v3 file not found: {raw} (pass --raw)")
    return actions, problems


def main(argv=None):
    a = parse_args(argv)
    if a.stage_worker:                                   # child process: run one stage, report, exit
        t0 = time.time()
        ok, err = True, None
        try:
            res = run_stage(a.stage_worker, a)
        except BaseException as e:                       # report, then fail the stage
            import traceback
            traceback.print_exc()
            res, ok, err = {}, False, f"{type(e).__name__}: {e}"
        payload = {"stage": a.stage_worker, "ok": ok, "error": err, "elapsed_seconds": round(time.time() - t0, 1),
                   "peak_rss_mb": peak_rss_mb(), "result": res}
        if a.stage_worker == "verify" and res and res.get("n_failed"):
            payload["ok"], payload["error"] = False, f"{res['n_failed']} verification check(s) FAILED"
        print(RESULT_TAG + json.dumps(payload, default=str), flush=True)
        sys.exit(0 if payload["ok"] else 3)

    t_start = time.time()
    actions, problems = plan(a)
    raw = Path(a.raw) if a.raw else paths.HDF5_DIR / paths.data_filename("1sec", "v3")
    n_rows = N_ROWS_REAL["1sec"]
    if raw.exists():
        try:
            from solete.h5io import h5_info
            n_rows = h5_info(raw, "DATA")["nrows"]
        except Exception:
            pass
    est = disk_estimate(n_rows)
    print(f"data folder : {paths.DATA_DIR}\nscratch     : {a.scratch}\nraw v3 file : {raw}{'' if raw.exists() else '  (not found)'}")
    print(f"stages      : " + ", ".join(f"{s}={actions[s]}" for s in a.stage_list))
    print(f"slice size  : {a.slice_days:g} days; rebuild check: {a.rebuild_check}")
    print("expected disk use (UPPER bound from {:,} rows, uncompressed; the real files are usually smaller):".format(n_rows))
    for k, v in est.items():
        print(f"    {k:<58s}{v / 1e9:>8.2f} GB")
    total = sum(est.values())
    print(f"    {'total':<58s}{total / 1e9:>8.2f} GB   (scratch part freed at the end unless --keep-intermediate)")
    need = {paths.DATA_DIR: 0, Path(a.scratch): 0}
    run = [s for s in a.stage_list if actions[s] == "run"]
    if "original" in run:
        need[paths.DATA_DIR] += est["original (h5)"]
    if "clean" in run:
        need[Path(a.scratch)] += est["scratch cleaned 1 s (h5)"]
    if "expand" in run:
        need[paths.DATA_DIR] += est["final 1 s (h5)"] + est["coarse files, scratch + final + parquet (h5 + parquet)"] // 3
    if "parquet" in run:
        need[paths.DATA_DIR] += est["parquet original"] + est["parquet 1 s"]
    if "verify" in run and a.rebuild_check == "full":
        need[paths.DATA_DIR] += est["scratch cleaned 1 s (h5)"] + est["final 1 s (h5)"]
    problems += check_disk(need)
    for p in problems:
        print("REFUSING:", p)
    if a.dry_run:
        print("\n--dry-run: nothing was written.")
        sys.exit(1 if problems else 0)
    if problems:
        sys.exit(2)

    if any(actions[s_] == "run" for s_ in a.stage_list if s_ in ("original", "clean", "resample", "expand", "parquet")):
        remove_stale_manifest("the release files are about to change")
    summary = {"version": "v4", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "data_dir": str(paths.DATA_DIR), "slice_days": a.slice_days, "rebuild_check": a.rebuild_check,
               "disk_upper_bound_gb": {k: round(v / 1e9, 2) for k, v in est.items()}, "stages": {}}
    failed = None
    for stage in a.stage_list:
        if actions[stage] == "skip":
            print(f"\n=== stage {stage}: skipped (outputs exist, --skip-existing) ===")
            summary["stages"][stage] = {"status": "skipped"}
            continue
        print(f"\n=== stage {stage} ===", flush=True)
        t0 = time.time()
        rc, res = run_child(stage, a)
        elapsed = round(time.time() - t0, 1)
        entry = {"status": "ok" if rc == 0 else "FAILED", "elapsed_seconds": elapsed,
                 "peak_rss_mb": (res or {}).get("peak_rss_mb"), "result": (res or {}).get("result")}
        if rc != 0:
            entry["error"] = (res or {}).get("error", f"exit code {rc}")
        summary["stages"][stage] = entry
        print(f"=== stage {stage}: {entry['status']} in {elapsed:.0f} s, peak RSS {entry['peak_rss_mb']} MB ===", flush=True)
        if rc != 0:
            failed = stage                                # the summary is still saved; no manifest after a failure
            remove_stale_manifest(f"stage {stage} failed")
            break

    summary["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    summary["total_elapsed_seconds"] = round(time.time() - t_start, 1)
    summary["failed_stage"] = failed
    summary["files"] = {k: {"bytes": Path(p).stat().st_size} for k, p in {**{f"{k}.h5": v for k, v in release_h5().items()},
                                                                      **{f"{k}.parquet": v for k, v in release_parquet().items()}}.items()
                        if Path(p).exists()}
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary_path().write_text(json.dumps(summary, indent=1, default=str))
    verify_rows = (summary["stages"].get("verify", {}).get("result") or {}).get("rows")
    if verify_rows:
        slim = dict(summary)
        slim["stages"] = {k: {kk: vv for kk, vv in v.items() if kk != "result"} for k, v in summary["stages"].items()}
        slim["verification"] = verify_rows
        print_report("build_release_summary", slim)
    else:
        print_report("build_release_summary", summary)
    if failed:
        print(f"BUILD FAILED at stage {failed}. Fix the cause and re-run with --skip-existing to resume.")
        sys.exit(3)
    if "manifest" in a.stage_list and not a.keep_intermediate and Path(a.scratch).exists():
        shutil.rmtree(a.scratch, ignore_errors=True)
        print(f"removed scratch folder {a.scratch}")
    print(f"Done in {summary['total_elapsed_seconds']:.0f} s. Summary: {summary_path()}")


if __name__ == "__main__":
    main()
