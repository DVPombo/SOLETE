# -*- coding: utf-8 -*-
"""Tests of the v4 release build (dataset/pipeline/*). Synthetic data only: they prove the code is
exact (sliced == whole-file, stages compose, files are what the docs say), not anything about the real data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataset" / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import clean_solete_1sec as cl  # noqa: E402
from _release_fixture import raw_defects, shuffled_raw, DAY  # noqa: E402
from solete.h5io import read_rows, h5_info  # noqa: E402


def _whole(df):
    out, rep = cl.clean_block(df, verbose=False)
    return out, cl.finish_report(rep)


@pytest.mark.parametrize("slice_rows", [7_000, 20_011, 60_000])
def test_sliced_cleaning_equals_whole_file(tmp_path, slice_rows):
    # defects straddle the natural cut points of THIS slicing
    cuts = range(slice_rows, 3 * DAY, slice_rows)
    df = raw_defects(3, cut_points=cuts)
    src = tmp_path / "in.h5"
    df.to_hdf(src, key="DATA", mode="w", format="table")
    whole, rep_whole = _whole(df)
    rep = cl.clean_file(src, tmp_path / "out.h5", slice_rows=slice_rows, verbose=False)
    got = read_rows(tmp_path / "out.h5")
    assert_frame_equal(got, whole, check_exact=True)
    # reports: the additive counts equal the whole-file ones
    for k in ("pressure_n_set_to_nan", "wind_dir_n_wrapped", "pressure_n_flagged_flatline_only",
              "pressure_n_days_with_real_value_remaining", "qc_flag_value_counts", "dropout", "glitch", "p_gaia"):
        assert rep[k] == rep_whole[k], k
    assert rep["n_slices"] > 1 or slice_rows >= 3 * DAY
    # every cut is a safe cut: no run-based rule region touches the two rows around it
    unsafe = cl.unsafe_row_mask(df)
    for b in rep["slice_boundaries_rows"]:
        assert not unsafe[b - 1] and not unsafe[b]


def test_cut_finder_skips_unsafe_regions(tmp_path):
    df = raw_defects(1, cut_points=[20_000])
    src = tmp_path / "in.h5"
    df.to_hdf(src, key="DATA", mode="w", format="table")
    c = cl.find_safe_cut(src, "DATA", 19_999, len(df), window=500)
    assert c >= 20_002          # the dropout rows 19_998..20_001 and the flatline around 20_000 are skipped
    assert cl.safe_boundaries(df.iloc[c - 1:c + 1]).all()


def test_output_columns_and_no_azel_flags(tmp_path):
    df = raw_defects(1)
    out, _ = _whole(shuffled_raw(df, with_az_el=True).sort_index())
    qc = [c for c in out.columns if c.endswith("_qc")]
    assert qc == [f"{c}_qc" for c in cl.QC_COLS] and len(qc) == 8
    assert "Azimuth[deg]_qc" not in out.columns and "Elevation[deg]_qc" not in out.columns
    assert list(out.columns[:11]) == list(df.columns) + ["Azimuth[deg]", "Elevation[deg]"]
    assert all(str(out[c].dtype) == "int8" for c in qc)


def test_unsorted_input_refused_without_in_memory(tmp_path):
    df = shuffled_raw(raw_defects(4), with_az_el=False)
    assert not df.index.is_monotonic_increasing
    src = tmp_path / "raw.h5"
    df.to_hdf(src, key="DATA", mode="w")      # fixed format, shuffled, like the real raw file
    with pytest.raises(ValueError, match="chronological"):
        cl.clean_file(src, tmp_path / "o.h5", verbose=False)


# ---- stage 0: the sorted _original file -------------------------------------------------------
import make_original as mo  # noqa: E402


@pytest.mark.parametrize("fmt", ["fixed", "table"])
def test_make_original_is_sorted_bit_identical_and_drops_azel(tmp_path, fmt):
    chrono = raw_defects(4)
    raw_df = shuffled_raw(chrono, with_az_el=True)
    raw = tmp_path / "raw.h5"
    raw_df.to_hdf(raw, key="DATA", mode="w", format=fmt)
    rep = mo.build_original(raw, tmp_path / "o.h5", chunk_rows=100_000, verbose=False)
    assert rep["n_rows_written"] == 4 * DAY and rep["dropped_columns"] == ["Azimuth[deg]", "Elevation[deg]"]
    assert mo.verify_original(raw, tmp_path / "o.h5", verbose=False)["identical"]
    got = read_rows(tmp_path / "o.h5")
    assert_frame_equal(got, chrono, check_exact=True, check_freq=False)    # chrono has the nine measured columns only
    assert h5_info(tmp_path / "o.h5")["format"] == "table"


def test_make_original_refuses_duplicates_and_overwrite(tmp_path):
    chrono = raw_defects(1)
    dup = pd.concat([chrono, chrono.iloc[:10]])
    raw = tmp_path / "dup.h5"
    dup.to_hdf(raw, key="DATA", mode="w")
    with pytest.raises(ValueError, match="duplicated"):
        mo.build_original(raw, tmp_path / "o.h5", verbose=False)
    ok = tmp_path / "ok.h5"
    chrono.to_hdf(ok, key="DATA", mode="w")
    mo.build_original(ok, tmp_path / "o2.h5", verbose=False)
    with pytest.raises(FileExistsError):
        mo.build_original(ok, tmp_path / "o2.h5", verbose=False)
    assert not list(tmp_path.glob("*.tmp"))


def test_verify_original_detects_a_changed_value(tmp_path):
    chrono = raw_defects(1)
    raw = tmp_path / "r.h5"
    chrono.to_hdf(raw, key="DATA", mode="w")
    mo.build_original(raw, tmp_path / "o.h5", verbose=False)
    import tables
    with tables.open_file(str(tmp_path / "o.h5"), "r+") as f:
        t = f.root.DATA.table
        arr = t.read(123, 124)
        arr["values_block_0"][0, 2] += 1.0
        t.modify_rows(123, 124, rows=arr)
    with pytest.raises(AssertionError, match="values differ"):
        mo.verify_original(raw, tmp_path / "o.h5", verbose=False)


# ---- stage 2: resampling ----------------------------------------------------------------------
import resample_solete as rs  # noqa: E402


def _cleaned_fixture(tmp_path, n_days=5, drop_day=None, boundary=False):
    df = raw_defects(n_days)
    if boundary:                                    # like the real raw file: one extra second at the next midnight
        df = pd.concat([df, df.iloc[[-1]].set_axis(df.index[-1:] + pd.Timedelta("1s"))])
    if drop_day is not None:                         # a whole missing day -> NaN buckets inside the grid
        d0 = df.index[0].normalize() + pd.Timedelta(days=drop_day)
        df = df[(df.index < d0) | (df.index >= d0 + pd.Timedelta(days=1))]
    cleaned, _ = _whole(df)
    path = tmp_path / "clean.h5"
    cleaned.to_hdf(path, key="DATA", mode="w", format="table")
    return cleaned, path


@pytest.mark.parametrize("slice_days,drop_day,boundary", [(1, None, False), (2, None, False), (2, 2, False), (3, 1, False),
                                                          (31, None, False), (2, None, True), (1, None, True)])
def test_sliced_resample_equals_whole_file(tmp_path, slice_days, drop_day, boundary):
    cleaned, path = _cleaned_fixture(tmp_path, 5, drop_day, boundary)
    got = rs.resample_file(path, "DATA", ["1min", "5min", "60min"], slice_days, verbose=False)
    for rule in ("1min", "5min", "60min"):
        want = rs.resample_dataframe(cleaned, rule)
        assert_frame_equal(got[rule], want, check_exact=True, check_freq=False), rule


def test_resample_has_no_azimuth_flags_and_circular_azimuth(tmp_path):
    cleaned, path = _cleaned_fixture(tmp_path, 2)
    out = rs.resample_file(path, "DATA", ["60min"], 1, verbose=False)["60min"]
    assert not any(c.startswith(("Azimuth[deg]_qc", "Elevation[deg]_qc")) for c in out.columns)
    assert "P_Gaia[kW]_qc_worst" in out.columns
    # azimuth buckets stay inside [-180, 180) and the bucket around solar midnight is not ~0 (= south)
    az = out["Azimuth[deg]"]
    assert az.min() >= -180 and az.max() < 180
    midnight = cleaned["Azimuth[deg]"].abs().groupby(cleaned.index.floor("60min")).min()
    assert (az.abs()[midnight > 150] > 120).all()


# ---- negative control: naive slicing really is different (so the exactness tests above have teeth) ----
def test_naive_cut_inside_a_run_changes_the_result():
    cut = 30_250                                   # inside the 997.5 flatline (30_000..30_500) and next to nothing else
    df = raw_defects(1)
    whole, _ = cl.clean_block(df, verbose=False)
    a, _ = cl.clean_block(df.iloc[:cut], verbose=False)
    b, _ = cl.clean_block(df.iloc[cut:], verbose=False)
    naive = pd.concat([a, b])
    assert not naive["Pressure[mbar]_qc"].equals(whole["Pressure[mbar]_qc"])      # two 250-sample halves escape the 300 limit
    unsafe = cl.safe_boundaries(df.iloc[cut - 1:cut + 1])
    assert not unsafe.all()                        # ... and the cut finder would have refused this cut


# ---- h5io against pandas -----------------------------------------------------------------------------
def test_h5io_reads_fixed_and_table_like_pandas(tmp_path):
    from solete.h5io import read_index
    df = raw_defects(1).iloc[:5000].copy()
    df["flag"] = (np.arange(len(df)) % 7).astype(np.int8)
    df["count"] = np.arange(len(df), dtype=np.int64)
    for fmt in ("fixed", "table"):
        p = tmp_path / f"{fmt}.h5"
        df.to_hdf(p, key="DATA", mode="w", format=fmt)
        assert h5_info(p)["format"] == fmt and h5_info(p)["nrows"] == len(df)
        assert read_index(p).equals(df.index)
        assert_frame_equal(read_rows(p, start=100, stop=4000), df.iloc[100:4000], check_freq=False)
        assert list(read_rows(p, columns=["flag", "WIND_DIR[deg]"]).columns) == ["flag", "WIND_DIR[deg]"]   # requested order, both formats
        assert_frame_equal(read_rows(p, start=4990, stop=99999), df.iloc[4990:], check_freq=False)


# ---- naming (implemented once, in solete/paths.py) --------------------------------------------------
def test_v4_file_names_follow_the_scheme():
    from solete import paths
    assert paths.release_stems() == ["SOLETE_Pombo_1sec_original_v4", "SOLETE_Pombo_1sec_v4", "SOLETE_Pombo_1min_v4",
                                     "SOLETE_Pombo_5min_v4", "SOLETE_Pombo_60min_v4"]
    assert paths.data_filename("1h", "v4") == "SOLETE_Pombo_60min_v4.h5"              # 1h stays an input alias
    assert paths.data_filename("60min", "v4", "parquet") == "SOLETE_Pombo_60min_v4.parquet"
    assert paths.data_filename("1sec", "v4", original=True) == "SOLETE_Pombo_1sec_original_v4.h5"
    assert paths.data_filename("1sec", "v3") == "SOLETE_Pombo_1sec.h5"               # v3 names unchanged
    with pytest.raises(ValueError):
        paths.data_filename("5min", "v4", original=True)


# ---- parquet spec -------------------------------------------------------------------------------------
def test_parquet_timestamp_is_pinned_ns_utc_with_metadata(tmp_path):
    import json
    import pyarrow.parquet as pq
    import export_parquet as ep
    cleaned, _ = _whole(raw_defects(1))
    cleaned.index = cleaned.index.as_unit("us")                         # what pandas 3 may hand over
    h5 = tmp_path / "SOLETE_Pombo_1sec_v4.h5"
    cleaned.to_hdf(h5, key="DATA", mode="w", format="table")
    out, n = ep.h5_to_parquet(h5, tmp_path / "x.parquet", chunk_rows=30_000, byte_stream_split=True, verbose=False)
    pf = pq.ParquetFile(out)
    assert str(pf.schema_arrow.field("timestamp").type) == "timestamp[ns, tz=UTC]" and pf.schema_arrow.names[0] == "timestamp"
    meta = json.loads(pf.schema_arrow.metadata[b"solete"])
    assert meta["version"] == "v4" and meta["timestamp_unit"] == "ns" and meta["resolution"] == "1sec"
    assert meta["site"]["latitude"] == 55.6867 and "0 = south" in meta["angle_convention"]
    assert meta["column_provenance"]["Azimuth[deg]"].startswith("recomputed by the pipeline")
    assert meta["column_provenance"]["WIND_DIR[deg]_qc"].startswith("pipeline flag")
    assert pf.metadata.row_group(0).num_rows <= ep.ROW_GROUP_SIZE and pf.metadata.row_group(0).column(1).statistics is not None
    assert ep.verify_roundtrip(h5, out, chunk_rows=40_000)["identical"]
    with pytest.raises(FileExistsError):
        ep.h5_to_parquet(h5, out, verbose=False)


def test_parquet_roundtrip_detects_a_difference(tmp_path):
    import export_parquet as ep
    cleaned, _ = _whole(raw_defects(1))
    h5 = tmp_path / "a.h5"
    cleaned.to_hdf(h5, key="DATA", mode="w", format="table")
    out, _ = ep.h5_to_parquet(h5, tmp_path / "a.parquet", verbose=False)
    changed = cleaned.copy()
    changed.iloc[777, 0] += 1.0
    changed.to_hdf(h5, key="DATA", mode="w", format="table")
    with pytest.raises(AssertionError):
        ep.verify_roundtrip(h5, out)


def test_disk_check_refuses_when_short(tmp_path):
    import build_release as br
    assert br.check_disk({tmp_path: 1}) == []
    assert br.check_disk({tmp_path: 10 ** 18})            # an exabyte is never free


# ---- the whole build, as the maintainer runs it (a subprocess, another working directory) -------------
def _run_build(data_dir, *args, cwd):
    import os
    import subprocess
    env = dict(os.environ, SOLETE_DATA_DIR=str(data_dir))
    return subprocess.run([sys.executable, str(ROOT / "dataset" / "pipeline" / "build_release.py"), *args],
                          capture_output=True, text=True, env=env, cwd=str(cwd))


def test_build_release_end_to_end_resume_refuse_and_tamper(tmp_path):
    import hashlib
    data, elsewhere = tmp_path / "data", tmp_path / "elsewhere"
    (data / "hdf5").mkdir(parents=True)
    elsewhere.mkdir()
    chrono = raw_defects(4, cut_points=[DAY + 43_200, 2 * DAY + 20_000])
    chrono = pd.concat([chrono, chrono.iloc[[-1]].set_axis(chrono.index[-1:] + pd.Timedelta("1s"))])   # boundary second, like the real raw file
    raw = data / "hdf5" / "SOLETE_Pombo_1sec.h5"
    shuffled_raw(chrono, with_az_el=True).to_hdf(raw, key="DATA", mode="w")
    raw_bytes = raw.read_bytes()

    dry = _run_build(data, "--dry-run", cwd=elsewhere)
    assert dry.returncode == 0 and "nothing was written" in dry.stdout and not (data / "parquet").exists()

    r = _run_build(data, "--slice-days", "1", "--rebuild-check", "full", cwd=elsewhere)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    table = r.stdout.split("=== stage verify")[1].split("=== stage manifest")[0]
    assert ", 0 FAILED," in table and "  FAIL  " not in table
    stems = ["SOLETE_Pombo_1sec_original_v4", "SOLETE_Pombo_1sec_v4", "SOLETE_Pombo_1min_v4", "SOLETE_Pombo_5min_v4", "SOLETE_Pombo_60min_v4"]
    for s in stems:
        assert (data / "hdf5" / f"{s}.h5").exists() and (data / "parquet" / f"{s}.parquet").exists()
    assert not list(data.rglob("*.tmp")) and not (data / "derived" / "build_v4").exists()   # scratch removed
    assert raw.read_bytes() == raw_bytes                                                      # input never modified
    assert (data / "build_summary_v4.json").exists() and (data / "SOLETE_Pombo_v4_RESAMPLING_METHODOLOGY.md").exists()
    # manifest and checksums describe what is on disk
    import json
    man = json.loads((data / "manifest.json").read_text())
    sums = dict(line.split("  ", 1)[::-1] for line in (data / "SHA256SUMS.txt").read_text().splitlines())
    assert len(man["files"]) == 12 and "SOLETE_Pombo_v4_RESAMPLING_METHODOLOGY.md" in sums and "figshare_README.txt" in sums
    fig = (data / "figshare_README.txt").read_text()
    assert "[[SIZES" not in fig and "hdf5/SOLETE_Pombo_60min_v4.h5" in fig and " MB" in fig       # real sizes, markers gone
    assert f"{4 * DAY + 1:,}" in fig and f"{4 * 24 + 1:,}" in fig                                    # real row counts, boundary second included
    for rel, digest in sums.items():
        assert hashlib.sha256((data / rel).read_bytes()).hexdigest() == digest
    one = read_rows(data / "hdf5" / "SOLETE_Pombo_1sec_v4.h5", stop=3)
    assert one.shape[1] == 28 and not any(c.startswith(("Azimuth[deg]_qc", "Elevation[deg]_qc")) for c in one.columns)
    assert list(read_rows(data / "hdf5" / "SOLETE_Pombo_1sec_original_v4.h5", stop=3).columns) == list(chrono.columns)
    assert h5_info(data / "hdf5" / "SOLETE_Pombo_1sec_v4.h5")["nrows"] == 4 * DAY + 1 and h5_info(data / "hdf5" / "SOLETE_Pombo_60min_v4.h5")["nrows"] == 4 * 24 + 1

    # a second run neither overwrites nor silently skips
    before = {p: p.stat().st_mtime_ns for p in (data / "hdf5").glob("*_v4.h5")}
    refused = _run_build(data, "--slice-days", "1", cwd=elsewhere)
    assert refused.returncode == 2 and "REFUSING" in refused.stdout
    assert before == {p: p.stat().st_mtime_ns for p in (data / "hdf5").glob("*_v4.h5")}
    resumed = _run_build(data, "--slice-days", "1", "--skip-existing", "--stages", "original,clean,resample,expand,parquet", cwd=elsewhere)
    assert resumed.returncode == 0 and resumed.stdout.count("skipped (outputs exist") == 5

    # tampering with a shipped file makes the verification fail loudly
    p60 = data / "hdf5" / "SOLETE_Pombo_60min_v4.h5"
    df = pd.read_hdf(p60)
    df.iloc[10, df.columns.get_loc("TempCell")] += 1.0
    df.to_hdf(p60, key="DATA", mode="w")
    bad = _run_build(data, "--stages", "verify", "--slice-days", "1", cwd=elsewhere)
    assert bad.returncode == 3 and "FAIL" in bad.stdout
    # a failed verification leaves no checksum list that could be published next to the bad files
    assert not (data / "SHA256SUMS.txt").exists() and not (data / "manifest.json").exists()
    assert "removed stale" in bad.stdout


def test_p_gaia_flag_is_never_ok_so_frac_flagged_is_one(tmp_path):
    """The documented 'degenerate case' (dataset/AGENTS.md, METHODOLOGY.md) is true for P_Gaia and only for it."""
    cleaned, _ = _whole(raw_defects(2))
    assert set(np.unique(cleaned["P_Gaia[kW]_qc"])) <= {7, 10} and (cleaned["P_Gaia[kW]_qc"] != 0).all()
    r = rs.resample_dataframe(cleaned, "60min")
    assert (r["P_Gaia[kW]_qc_frac_flagged"] == 1.0).all()
    assert not any("Azimuth" in c and c.endswith("_qc") for c in cleaned.columns)


def test_grid_check_derives_row_counts_and_accepts_the_boundary_second(tmp_path):
    import release_verify as rv
    # the real span with and without the boundary second: counts follow from the span
    a, b = pd.Timestamp("2018-06-01"), pd.Timestamp("2019-08-31 23:59:59")
    assert [rv.expected_rows(r, a, b) for r in ("1sec", "1min", "5min", "60min")] == [39_484_800, 658_080, 131_616, 10_968]
    b2 = pd.Timestamp("2019-09-01 00:00:00")
    assert [rv.expected_rows(r, a, b2) for r in ("1sec", "1min", "5min", "60min")] == [39_484_801, 658_081, 131_617, 10_969]
    assert rv.is_real_span(a, b) and rv.is_real_span(a, b2) and not rv.is_real_span(a, pd.Timestamp("2019-09-02"))
    # a file with the boundary second passes check_grid; a file with a missing second does not
    cleaned, path = _cleaned_fixture(tmp_path, 2, boundary=True)
    assert "gap-free" in rv.check_grid(path, "1sec")
    gap = cleaned.drop(cleaned.index[1000])
    gp = tmp_path / "gap.h5"
    gap.to_hdf(gp, key="DATA", mode="w", format="table")
    with pytest.raises(AssertionError):
        rv.check_grid(gp, "1sec")


def test_compression_chooser_requires_a_real_gain():
    import export_parquet as ep
    base = {"compression": "zstd", "byte_stream_split": False, "rows": 10}
    flat = [dict(base, level=3, size_bytes=1000, write_seconds=1.0), dict(base, level=9, size_bytes=996, write_seconds=1.1),
            dict(base, level=15, size_bytes=993, write_seconds=2.4), dict(base, level=3, byte_stream_split=True, size_bytes=1076, write_seconds=.5)]
    assert ep.choose_settings(flat)["compression_level"] == 3                       # 0.7 % is not worth 2.4x the time (your real trial)
    better = flat[:2] + [dict(base, level=15, size_bytes=900, write_seconds=2.4)]
    assert ep.choose_settings(better)["compression_level"] == 15
    slow = flat[:1] + [dict(base, level=15, size_bytes=500, write_seconds=9.0)]       # beyond 4x: refused
    assert ep.choose_settings(slow)["compression_level"] == 3
