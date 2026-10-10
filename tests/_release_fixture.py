# -*- coding: utf-8 -*-
"""Small raw-like 1-second files with every defect the cleaning rules look for, placed where slicing is
most likely to go wrong (across the slice cut points the tests use). SYNTHETIC: proves code, not data."""
import numpy as np
import pandas as pd

from solete.synthetic import synthetic_solete

WD, WS, HUM, PRES = "WIND_DIR[deg]", "WIND_SPEED[m1s]", "HUMIDITY[%]", "Pressure[mbar]"
TEMP, GHI, POA, PG = "TEMPERATURE[degC]", "GHI[kW1m2]", "POA Irr[kW1m2]", "P_Gaia[kW]"
DAY = 86400


def raw_defects(n_days=3, start="2019-01-15", seed=1, cut_points=(), cut_width=12):
    """Chronological frame with defects; `cut_points` are row positions where extra defects straddle."""
    df = synthetic_solete(start=start, periods=n_days * DAY, seed=seed)
    df = df[[c for c in df.columns if c not in ("Azimuth[deg]", "Elevation[deg]")]].copy()
    n = len(df)
    rng = np.random.default_rng(seed)
    col = {c: df[c].to_numpy().copy() for c in df.columns}
    col[WD][1000:1010] = 361.5 + np.arange(10)           # above 360
    col[WD][5000] = -3.0
    col[PRES][:] = 1000.0                                  # sentinel stretches ...
    r0 = (DAY if n_days > 1 else 0) + 60_000
    col[PRES][r0: r0 + 5000] = 1013.25 + np.cumsum(rng.normal(0, 0.01, 5000))  # real stretch
    col[PRES][30_000:30_500] = 997.5                       # non-round flatline plateau (>= 300)
    col[PRES][40_000:40_100] = 998.5                       # short identical run (< 300): stays real
    col[PRES][50_000:50_003] = np.nan
    col[WS][10_000:10_003] = 0.0; col[HUM][10_000:10_003] = 0.0       # short dropout (fixed)
    col[WS][20_000:20_020] = 0.0; col[HUM][20_000:20_020] = 0.0       # long dropout (untreated)
    col[TEMP][15_000] = -40.1                              # single glitch
    col[TEMP][16_000:16_010] = 99.0                        # long glitch (NaN)
    col[GHI][17_000:17_002] = 9.0                          # short glitch
    col[POA][18_000] = np.nan
    col[HUM][19_000:19_003] = 1.7                          # humidity glitch
    for c in cut_points:                                   # defects straddling the cut points
        c = int(c)
        if cut_width < c < n - cut_width:
            col[WS][c - 2:c + 2] = 0.0; col[HUM][c - 2:c + 2] = 0.0     # dropout across the cut
            col[TEMP][c + 5:c + 7] = 80.0                              # glitch just after
            col[GHI][c - 7:c - 5] = 5.0                                # glitch just before
            col[PRES][c - 400:c + 400] = 1001.2                        # flatline across the cut
    return pd.DataFrame(col, index=df.index)


def shuffled_raw(df, with_az_el=True, seed=2):
    """Daily blocks in shuffled order, plus (optionally) the published Azimuth/Elevation columns."""
    days = [g for _, g in df.groupby(df.index.normalize())]
    order = np.random.default_rng(seed).permutation(len(days))
    out = pd.concat([days[i] for i in order])
    if with_az_el:
        out["Azimuth[deg]"] = 0.0
        out["Elevation[deg]"] = 0.0
    return out
