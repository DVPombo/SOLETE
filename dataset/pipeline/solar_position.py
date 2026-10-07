"""
solar_position.py  (v2 -- patched after the first real run; see CHANGES below)

Compute solar azimuth/elevation from timestamp + site coordinates, and validate
against the one populated day (2019-01-16) in the published hourly file.

CHANGES vs v1 (all found by running v1 against the real hourly file):
  1. BUG: v1 returned a tz-aware (UTC) index, the hourly file's index is tz-naive,
     so `reindex` matched nothing and every recomputed value was NaN. v2 returns a
     DataFrame indexed exactly like the input timestamps.
  2. TIMEZONE (revised after the GHI check, followup_diagnostics section D):
     - The LEGACY Azimuth/Elevation values on 2019-01-16 behave as if the
       timestamps were UTC+1 (their nonzero window is centred on 12:21:20, solar
       noon in CET).
     - But the MEASURED irradiance says the file's timestamps are UTC, all year:
       the GHI-weighted daily centroid minus computed solar noon (UTC) is about
       -0.1 h in every month, in summer as well as winter -- neither +1 (CET) nor
       +2 (CEST).  The original release's documentation does not state the
       timezone (checked by the maintainer); the wording in v1's warning that
       claimed otherwise was unsupported.
     - So new values are computed with DEFAULT_UTC_OFFSET_HOURS = 0, and the
       validation reports which offset the LEGACY values best fit (+1), which is
       evidence the legacy values were computed assuming local time.
  3. AZIMUTH CONVENTION: the published values are referenced to SOUTH (0 = S,
     east negative), pvlib's are referenced to NORTH (0 = N, clockwise). v1
     compared them directly (off by 180 deg). v2 adds `azimuth_south`.
  4. The published hourly value is a MEAN over the bucket [T, T+1h), not an
     instantaneous value; v2 validates against bucket means (1-minute sampling).
  5. Validation now tries several UTC offsets and reports all of them, so the
     timezone choice is evidence-based instead of assumed.

Usage:
    python solar_position.py SOLETE_Pombo_60min.h5
Requires: pip install pvlib
"""
import numpy as np
import pandas as pd

from solete_report import print_report

SITE_LATITUDE = 55.6867
SITE_LONGITUDE = 12.0985
SITE_ALTITUDE_M = 10
DEFAULT_UTC_OFFSET_HOURS = 0.0   # file timestamps are UTC -- see CHANGES #2


def _to_utc(index, utc_offset_hours):
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        return (idx - pd.Timedelta(hours=utc_offset_hours)).tz_localize("UTC")
    return idx.tz_convert("UTC")


def compute_solar_position(
    timestamps,
    latitude=SITE_LATITUDE,
    longitude=SITE_LONGITUDE,
    altitude=SITE_ALTITUDE_M,
    utc_offset_hours=DEFAULT_UTC_OFFSET_HOURS,
):
    """
    Instantaneous solar position at each timestamp (naive timestamps are read as
    UTC + utc_offset_hours). Returned index == input index.
    Columns: azimuth (pvlib, 0=N clockwise), azimuth_south (0=S, east negative --
    the convention of the published file), elevation (apparent, refraction-
    corrected), elevation_geometric, zenith (apparent).
    NREL SPA via pvlib.
    """
    import pvlib

    original_index = pd.DatetimeIndex(timestamps)
    utc = _to_utc(original_index, utc_offset_hours)
    pos = pvlib.solarposition.get_solarposition(utc, latitude, longitude, altitude=altitude)
    az = pos["azimuth"].to_numpy()
    return pd.DataFrame(
        {
            "azimuth": az,
            "azimuth_south": az - 180.0,
            "elevation": pos["apparent_elevation"].to_numpy(),
            "elevation_geometric": pos["elevation"].to_numpy(),
            "zenith": pos["apparent_zenith"].to_numpy(),
        },
        index=original_index,
    )


def compute_solar_position_bulk(
    timestamps,
    latitude=SITE_LATITUDE,
    longitude=SITE_LONGITUDE,
    altitude=SITE_ALTITUDE_M,
    utc_offset_hours=DEFAULT_UTC_OFFSET_HOURS,
    chunk_rows=2_000_000,
    verbose=True,
):
    """
    Same as compute_solar_position, but for a long index processed in chunks
    (bounds peak memory/pvlib overhead on a 39M-row 1-second index). Prints
    progress. Returns one concatenated DataFrame indexed like `timestamps`.
    """
    idx = pd.DatetimeIndex(timestamps)
    n = len(idx)
    parts = []
    for i in range(0, n, chunk_rows):
        parts.append(
            compute_solar_position(
                idx[i:i + chunk_rows], latitude, longitude, altitude, utc_offset_hours
            )
        )
        if verbose:
            print(f"  solar position {min(i + chunk_rows, n):,} / {n:,}")
    return pd.concat(parts)


def bucket_mean_solar_position(bucket_starts, period="1h", step="1min",
                               utc_offset_hours=DEFAULT_UTC_OFFSET_HOURS, **site):
    """Mean of instantaneous positions over each [T, T+period) bucket (label = start)."""
    starts = pd.DatetimeIndex(bucket_starts)
    half = pd.Timedelta(step) / 2   # sample at sub-interval midpoints (avoids a half-step bias)
    offs = pd.timedelta_range(start=half, end=pd.Timedelta(period) - half, freq=step)
    all_t = pd.DatetimeIndex((starts.values[:, None] + offs.values[None, :]).ravel())
    pos = compute_solar_position(all_t, utc_offset_hours=utc_offset_hours, **site)
    n = len(offs)
    out = {c: pos[c].to_numpy().reshape(len(starts), n).mean(axis=1) for c in pos.columns}
    return pd.DataFrame(out, index=starts)


def validate_against_known_day(
    df_original,
    df_recomputed=None,          # kept only so old calls still work; ignored in v2
    known_day="2019-01-16",
    azimuth_col="Azimuth[deg]",
    elevation_col="Elevation[deg]",
    candidate_offsets=(-1.0, 0.0, 1.0, 2.0),
    chosen_offset=DEFAULT_UTC_OFFSET_HOURS,
    min_real_elevation=1.0,
    period="1h",
):
    """
    Compares the real published values on the known day with bucket-mean pvlib
    values, for several UTC offsets. Only buckets whose real elevation exceeds
    `min_real_elevation` are compared (edge buckets are partly zero-clipped in the
    published file). Residual = real - recomputed.
    """
    if df_original.index.tz is not None:
        raise ValueError("Expected a tz-naive index (as in the published file).")
    day = pd.Timestamp(known_day)
    idx = df_original.index
    real = df_original.loc[(idx >= day) & (idx < day + pd.Timedelta(days=1)), [azimuth_col, elevation_col]]
    usable = real[real[elevation_col] > min_real_elevation]
    if usable.empty:
        print_report("solar_position_validation_v2", {"error": "no usable buckets", "known_day": known_day})
        return None

    summary, detail = [], {}
    for off in sorted(set(candidate_offsets) | {chosen_offset}):
        bm = bucket_mean_solar_position(usable.index, period=period, utc_offset_hours=off)
        az_res = usable[azimuth_col].to_numpy() - bm["azimuth_south"].to_numpy()
        el_res = usable[elevation_col].to_numpy() - bm["elevation"].to_numpy()
        row = {
            "utc_offset_hours": off,
            "mean_az_residual": float(az_res.mean()), "std_az_residual": float(az_res.std()),
            "mean_abs_az_residual": float(np.abs(az_res).mean()), "max_abs_az_residual": float(np.abs(az_res).max()),
            "mean_el_residual": float(el_res.mean()), "std_el_residual": float(el_res.std()),
            "mean_abs_el_residual": float(np.abs(el_res).mean()), "max_abs_el_residual": float(np.abs(el_res).max()),
        }
        row["score_mean_abs_az_plus_el"] = row["mean_abs_az_residual"] + row["mean_abs_el_residual"]
        summary.append(row)
        detail[off] = pd.DataFrame(
            {"real_azimuth_south": usable[azimuth_col].to_numpy(), "recomputed_azimuth_south": bm["azimuth_south"].to_numpy(),
             "az_residual": az_res, "real_elevation": usable[elevation_col].to_numpy(),
             "recomputed_elevation": bm["elevation"].to_numpy(), "el_residual": el_res},
            index=usable.index)

    summary.sort(key=lambda r: r["score_mean_abs_az_plus_el"])
    best = summary[0]

    print("Candidate UTC offsets (best first); residual = real - recomputed, degrees:")
    print(pd.DataFrame(summary).round(3).to_string(index=False))
    print(f"\nDetail for chosen offset {chosen_offset:+.1f} h:")
    print(detail[chosen_offset].round(3).to_string())
    print()

    print_report(
        "solar_position_validation_v2",
        {
            "known_day": known_day, "site_latitude": SITE_LATITUDE, "site_longitude": SITE_LONGITUDE,
            "site_altitude_m": SITE_ALTITUDE_M, "n_buckets_compared": len(usable),
            "azimuth_convention_compared": "real vs pvlib_azimuth-180 (0=south, east negative)",
            "chosen_utc_offset_hours": chosen_offset,
            "best_utc_offset_hours": best["utc_offset_hours"],
            "legacy_values_consistent_with_chosen_offset": best["utc_offset_hours"] == chosen_offset,
            "legacy_values_best_fit_utc_offset_hours": best["utc_offset_hours"],
            "residual_at_best_offset_is_constant": bool(
                best["std_az_residual"] < 0.1 and best["std_el_residual"] < 0.1
                and (abs(best["mean_az_residual"]) > 0.5 or abs(best["mean_el_residual"]) > 0.2)),
            "best_offset_residual_mean_az_el": [best["mean_az_residual"], best["mean_el_residual"]],
            "per_offset_summary": summary,
            "row_by_row_chosen_offset": detail[chosen_offset].reset_index().rename(columns={"index": "bucket"}).to_dict(orient="records"),
        },
    )
    return detail[chosen_offset]


if __name__ == "__main__":
    import sys

    hourly_path = sys.argv[1] if len(sys.argv) > 1 else "SOLETE_Pombo_60min.h5"
    df = pd.read_hdf(hourly_path, key="DATA")
    validate_against_known_day(df)
