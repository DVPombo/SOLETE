"""
followup_diagnostics.py -- second-round checks, written after the first real run.

Usage (Spyder):
    %run followup_diagnostics.py SOLETE_Pombo_1sec.h5 SOLETE_Pombo_60min.h5 SOLETE_Pombo_60min_v4.h5

Prints six report blocks (paste them all back, BEGIN/END lines included):
  followup_A_index_order        why the 1-s index is non-monotonic; is the grid complete?
  followup_A2_post_sort_runs    constant-run / zero-run stats recomputed in chronological order
  followup_B_wind_dir           raw values >360; can ANY averaging produce the hourly >360 values?
  followup_C_wind_power         independent (numpy) re-check of the wind-vs-power consistency block
  followup_D_timezone_from_ghi  implied UTC offset per month (fixed UTC+1, or +1/+2 with DST?)
  followup_E_other_oddities     humidity>1, temperature -40.1, sentinel pressure, POA outliers, ...

Memory: the 1-s file is sorted chronologically, which temporarily needs ~2x the
dataframe size. Run `%reset -f` first. Each section is wrapped so that an error in
one prints a traceback inside its own block and the rest still run.
"""
import sys
import traceback

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "pipeline"))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # repo root: gives `import solete.paths`
from solete.paths import resolve_input
from solete_report import print_report

WD, WS, PG = "WIND_DIR[deg]", "WIND_SPEED[m1s]", "P_Gaia[kW]"
GHI, POA = "GHI[kW1m2]", "POA Irr[kW1m2]"
TEMP, HUM, PRES = "TEMPERATURE[degC]", "HUMIDITY[%]", "Pressure[mbar]"
SITE_LON = 12.0985
CUTIN, CUTOUT, POWER_FLOOR = 3.5, 25.0, 0.01


# ----------------------------------------------------------------- helpers
def safe(name, fn, *args, **kwargs):
    try:
        out = fn(*args, **kwargs)
    except Exception as exc:  # keep going, show the problem in the report
        out = {"error": repr(exc), "traceback": traceback.format_exc()}
    print_report(name, out)


def bool_runs(b):
    """Start/end (inclusive) positions of runs of True in a boolean array."""
    d = np.diff(np.concatenate(([0], np.asarray(b).astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1


def longest_run(a):
    if len(a) == 0:
        return 0
    change = np.flatnonzero(a[1:] != a[:-1])
    edges = np.concatenate(([-1], change, [len(a) - 1]))
    return int(np.diff(edges).max())


def top_days(index, mask, top=10):
    ts = index[np.asarray(mask)]
    if len(ts) == 0:
        return {}
    vc = ts.normalize().value_counts().head(top)
    return {str(k.date()): int(v) for k, v in vc.items()}


# ------------------------------------------------------------------- A
def check_index_order(df):
    idx = df.index
    out = {"n_rows": int(len(idx)), "is_monotonic_increasing": bool(idx.is_monotonic_increasing)}
    ns = idx.values.astype("datetime64[ns]").astype("int64")
    out["all_timestamps_whole_seconds"] = bool((ns % 1_000_000_000 == 0).all())
    sec = ns // 1_000_000_000
    step = np.diff(sec)
    breaks = np.flatnonzero(step != 1)
    out["n_steps_not_plus_1s"] = int(len(breaks))
    out["n_backward_steps"] = int((step < 0).sum())
    span_n = int((idx.max() - idx.min()) / pd.Timedelta("1s")) + 1
    out["span_start"], out["span_end"] = str(idx.min()), str(idx.max())
    out["rows_expected_for_complete_1s_grid"] = span_n
    out["n_unique_timestamps"] = int(idx.nunique())
    out["complete_1s_grid_when_sorted"] = bool(span_n == idx.nunique() == len(idx))
    # file-order blocks between breaks
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(idx) - 1]))
    blocks = [{"file_pos_start": int(s), "n_rows": int(e - s + 1),
               "first_ts": str(idx[s]), "last_ts": str(idx[e])}
              for s, e in zip(starts[:40], ends[:40])]
    out["n_blocks_in_file_order"] = int(len(starts))
    out["first_40_blocks_in_file_order"] = blocks
    return out


def post_sort_runs(df):
    out = {"longest_constant_run_rows_chronological": {c: longest_run(df[c].to_numpy()) for c in df.columns}}
    p = df[PG].to_numpy()
    zs, ze = bool_runs(np.abs(p) < POWER_FLOOR)
    order = np.argsort(-(ze - zs))[:8]
    out["P_Gaia_n_zero_runs"] = int(len(zs))
    out["P_Gaia_longest_zero_runs"] = [
        {"start": str(df.index[zs[i]]), "end": str(df.index[ze[i]]),
         "days": round(float((ze[i] - zs[i] + 1) / 86400), 3)} for i in order]
    return out


# ------------------------------------------------------------------- B
def wind_dir_diag(df, orig_hourly, fresh_hourly):
    wd = df[WD]
    a = wd.to_numpy()
    over = a > 360
    out = {"raw_max": float(a.max()), "raw_n_negative": int((a < 0).sum()), "raw_n_over_360": int(over.sum()),
           "raw_n_exactly_360": int((a == 360).sum())}
    if over.any():
        ts = wd.index[over]
        out["raw_over_360_first_ts"], out["raw_over_360_last_ts"] = str(ts.min()), str(ts.max())
        out["raw_over_360_n_days"] = int(ts.normalize().nunique())
        out["raw_over_360_top_days"] = top_days(wd.index, over, 15)
        v = a[over]
        out["raw_over_360_value_quantiles"] = {k: float(np.quantile(v, q)) for k, q in
                                               [("min", 0), ("p25", .25), ("median", .5), ("p75", .75), ("max", 1)]}
        # Are they "true angle + 360"? compare (v-360) with the median of nearby in-range samples
        dev = []
        for p in np.flatnonzero(over)[:5000]:
            lo, hi = max(0, p - 30), min(len(a), p + 31)
            nb = np.concatenate((a[lo:p], a[p + 1:hi]))
            nb = nb[nb <= 360]
            if len(nb):
                dev.append(abs(((a[p] - 360) - np.median(nb) + 180) % 360 - 180))
        dev = np.array(dev)
        out["unwrap_test"] = {
            "n_tested": int(len(dev)),
            "median_abs_dev_of_(v-360)_from_neighbour_median_deg": float(np.median(dev)) if len(dev) else None,
            "fraction_within_20deg": float((dev <= 20).mean()) if len(dev) else None,
            "reading": "fraction near 1 => values look like true angle + 360 (unwrapped vane); near 0 => junk/glitch values",
        }

    hourly = wd.resample("1h").agg(["min", "max", "mean"])
    hourly["n_over_360"] = (wd > 360).resample("1h").sum()
    j = hourly.join(orig_hourly[WD].rename("orig"), how="inner").join(fresh_hourly[WD].rename("circular"), how="inner")
    d = (j["orig"] - j["mean"]).abs()
    out["hourly_original_vs_plain_arithmetic_mean_of_raw"] = {
        "n_hours": int(len(j)), "n_within_1e-6": int((d <= 1e-6).sum()), "n_within_1deg": int((d <= 1).sum()),
        "max_abs_diff": float(d.max())}
    oh = j[j["orig"] > 360]
    out["original_hourly_n_over_360"] = int(len(oh))
    out["original_hourly_n_over_360_where_raw_hour_has_no_value_over_360"] = int((oh["n_over_360"] == 0).sum())
    out["original_hourly_n_over_360_exceeding_raw_hour_max"] = int((oh["orig"] > oh["max"] + 1e-6).sum())
    out["all_hours_original_above_raw_hour_max"] = int((j["orig"] > j["max"] + 1e-6).sum())
    out["all_hours_original_below_raw_hour_min"] = int((j["orig"] < j["min"] - 1e-6).sum())
    out["reading"] = ("any mean of raw samples must lie within [raw_min, raw_max] of that hour; "
                      "'exceeding_raw_hour_max' > 0 proves the original hourly value did NOT come from averaging this raw data")
    out["examples_original_over_360"] = oh.head(12).rename_axis("hour").reset_index().to_dict(orient="records")
    return out


# ------------------------------------------------------------------- C
def wind_power_diag(df):
    idx = df.index
    s, p = df[WS].to_numpy(), df[PG].to_numpy()
    nz = np.abs(p) >= POWER_FLOOR
    bands = {"speed<=3.5": s <= CUTIN, "3.5<speed<=25": (s > CUTIN) & (s <= CUTOUT), "speed>25": s > CUTOUT}
    out = {"power_floor_kW": POWER_FLOOR, "n_rows_power_nonzero_total": int(nz.sum()),
           "crosstab_speed_band_vs_power": {k: {"n_rows": int(m.sum()), "n_power_nonzero": int((m & nz).sum()),
                                                "n_power_near_zero": int((m & ~nz).sum()),
                                                "pct_near_zero": float(100 * (m & ~nz).sum() / max(m.sum(), 1))}
                                            for k, m in bands.items()}}
    hi = s > CUTOUT
    out["rows_above_cutout"] = [{"ts": str(t), "wind": float(w), "power": float(q)}
                                for t, w, q in zip(idx[hi][:50], s[hi][:50], p[hi][:50])]
    out["nonzero_power_seconds_by_day"] = top_days(idx, nz, 20)
    starts, ends = bool_runs(nz)
    out["n_nonzero_power_segments"] = int(len(starts))
    order = np.argsort(-(ends - starts))[:15]
    out["longest_nonzero_power_segments"] = [
        {"start": str(idx[starts[i]]), "end": str(idx[ends[i]]), "n_rows": int(ends[i] - starts[i] + 1),
         "mean_power": float(p[starts[i]:ends[i] + 1].mean()), "max_power": float(p[starts[i]:ends[i] + 1].max()),
         "mean_wind": float(s[starts[i]:ends[i] + 1].mean())} for i in order]
    # power curve on the days where the turbine produced anything
    active_days = set(idx[nz].normalize().unique())
    on_active = np.asarray(idx.normalize().isin(active_days))
    curve = pd.DataFrame({"bin": np.floor(s[on_active]), "nz": nz[on_active], "p": p[on_active]}).groupby("bin").agg(
        n=("p", "size"), frac_nonzero=("nz", "mean"), mean_power=("p", "mean"))
    out["power_vs_wind_1ms_bins_on_active_days_only"] = curve.round(4).reset_index().to_dict(orient="records")
    # windy days with no production at all
    daily = pd.DataFrame({"ws": s, "nz": nz.astype(np.float32)}, index=idx).resample("1D").mean()
    windy_idle = daily[(daily["ws"] >= 6) & (daily["nz"] == 0)]
    out["n_days"] = int(len(daily))
    out["n_days_mean_wind_ge_6_and_zero_power_all_day"] = int(len(windy_idle))
    out["windiest_zero_power_days"] = {str(k.date()): round(float(v), 2)
                                       for k, v in windy_idle["ws"].sort_values(ascending=False).head(10).items()}
    return out


# ------------------------------------------------------------------- D
def tz_from_ghi(df, lon=SITE_LON):
    import pvlib
    g = df[GHI].resample("1min").mean()
    day = g.index.normalize()
    hrs = (g.index - day).total_seconds().to_numpy() / 3600.0
    w = g.to_numpy()
    tmp = pd.DataFrame({"day": day, "w": w, "wh": w * hrs})
    agg = tmp.groupby("day").agg(w=("w", "sum"), wh=("wh", "sum"))
    agg["kwh_m2"] = agg["w"] / 60.0
    agg = agg[agg["kwh_m2"] > 0.8]                        # skip very dark days
    agg["centroid_h"] = agg["wh"] / agg["w"]
    eot = pvlib.solarposition.equation_of_time_spencer71(agg.index.dayofyear.to_numpy())
    agg["solar_noon_utc_h"] = 12.0 - lon / 15.0 - np.asarray(eot) / 60.0
    agg["implied_utc_offset_h"] = agg["centroid_h"] - agg["solar_noon_utc_h"]
    m = agg.groupby(agg.index.to_period("M"))["implied_utc_offset_h"].agg(
        n_days="size", median="median", p25=lambda x: x.quantile(.25), p75=lambda x: x.quantile(.75))
    return {"method": "GHI-weighted centroid time of day vs computed solar noon (UTC); cloudy days skipped",
            "reading": "~+1.0 all year => fixed CET (UTC+1); ~+1.0 in winter and ~+2.0 in summer => local time with DST",
            "per_month": [{"month": str(k), **{c: round(float(v), 3) for c, v in r.items()}} for k, r in m.iterrows()]}


# ------------------------------------------------------------------- E
def other_oddities(df):
    idx = df.index
    t, h, ws = df[TEMP].to_numpy(), df[HUM].to_numpy(), df[WS].to_numpy()
    pr, poa, ghi = df[PRES].to_numpy(), df[POA].to_numpy(), df[GHI].to_numpy()
    out = {}
    out["temperature_le_-30"] = {"n": int((t <= -30).sum()), "top_days": top_days(idx, t <= -30)}
    out["humidity_gt_1"] = {"n": int((h > 1).sum()), "top_days": top_days(idx, h > 1),
                            "value_counts_top5": {str(k): int(v) for k, v in pd.Series(h[h > 1]).value_counts().head(5).items()}}
    zw, zh = ws == 0, h == 0
    out["wind_speed_zero_vs_humidity_zero"] = {"n_ws_zero": int(zw.sum()), "n_hum_zero": int(zh.sum()),
                                               "n_both_zero_same_second": int((zw & zh).sum()),
                                               "top_days_ws_zero": top_days(idx, zw)}
    sent = np.isin(pr, [1000.0, 2000.0, 3000.0])
    out["pressure_sentinel_values"] = {"n_sentinel_rows": int(sent.sum()), "pct": float(100 * sent.mean()),
                                       "n_days_with_any_real_pressure": int(idx[~sent].normalize().nunique()),
                                       "first_real_ts": str(idx[~sent].min()) if (~sent).any() else None,
                                       "last_real_ts": str(idx[~sent].max()) if (~sent).any() else None}
    out["poa_gt_1.6_kW_m2"] = {"n": int((poa > 1.6).sum()), "max": float(poa.max()), "top_days": top_days(idx, poa > 1.6)}
    out["ghi_gt_1.4_kW_m2"] = {"n": int((ghi > 1.4).sum()), "max": float(ghi.max())}
    out["wind_speed_ge_25"] = {"n": int((ws >= 25).sum()), "max": float(ws.max()), "top_days": top_days(idx, ws >= 25)}
    return out


# ------------------------------------------------------------------- main
def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return
    p1s, p60, pfresh = [str(resolve_input(a)) for a in sys.argv[1:4]]
    print(f"Loading {p1s} ...")
    df = pd.read_hdf(p1s, key="DATA")
    safe("followup_A_index_order", check_index_order, df)
    if not df.index.is_monotonic_increasing:
        print("Sorting chronologically (needs extra RAM) ...")
        df = df.sort_index()
    safe("followup_A2_post_sort_runs", post_sort_runs, df)
    orig = pd.read_hdf(p60, key="DATA")
    fresh = pd.read_hdf(pfresh, key="DATA")
    safe("followup_B_wind_dir", wind_dir_diag, df, orig, fresh)
    safe("followup_C_wind_power", wind_power_diag, df)
    safe("followup_D_timezone_from_ghi", tz_from_ghi, df)
    safe("followup_E_other_oddities", other_oddities, df)


if __name__ == "__main__":
    main()
