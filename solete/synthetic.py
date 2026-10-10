# -*- coding: utf-8 -*-
"""
solete/synthetic.py -- a deterministic, SYNTHETIC SOLETE-like table, for tests and for the
expansion diagnostics (scripts/expansion_checks.py) when the real 1-second file is not at hand.

It is NOT data and proves nothing about the real dataset: it only has the right columns, units,
plausible magnitudes, a day/night cycle, cloud-like variability, measured P_Solar that sometimes sits
below the model (curtailment-like blocks) and a mostly-zero P_Gaia. Any number computed from it is a
statement about this generator.

Pure NumPy/pandas (no CoolProp), cheap enough for a month of 1-second rows.
"""

import numpy as np
import pandas as pd

from .physics import pv_model_arrays

_SPS = {'1s': 1, '1sec': 1, '1min': 60, '5min': 300, '60min': 3600, '1h': 3600}


def synthetic_solete(start='2019-06-01', periods=86400, freq='1s', seed=0, with_az_el=False):
    """Return a UTC-naive DataFrame with the v4 measured columns (no flags, no model columns).

    The underlying signals are built per second and smoothed to the requested step by averaging,
    so tables at different `freq` made with the same `seed`/span describe the same "weather"."""
    from .params import import_PV_WT_data
    step = _SPS[freq] if freq in _SPS else int(pd.Timedelta(freq).total_seconds())
    n = periods * step
    rng = np.random.default_rng(seed)
    idx1 = pd.date_range(start, periods=n, freq='1s')
    sec = (idx1.hour.to_numpy() * 3600 + idx1.minute.to_numpy() * 60 + idx1.second.to_numpy()).astype(np.float64)
    doy = idx1.dayofyear.to_numpy().astype(np.float64)

    def smooth_noise(scale_s, size):
        k = max(int(scale_s), 1)
        m = size // k + 2
        base = rng.standard_normal(m)
        return np.interp(np.arange(size) / k, np.arange(m), base)

    day = np.clip(np.sin(np.pi * (sec - 4.5 * 3600) / (15 * 3600)), 0, None)       # ~04:30..19:30 UTC
    clear = 1.05 * day ** 1.2 * (0.9 + 0.1 * np.cos(2 * np.pi * (doy - 172) / 365))  # kW/m2 plane of array
    cloud = np.clip(0.75 + 0.35 * smooth_noise(900, n) + 0.12 * smooth_noise(20, n), 0.05, 1.15)
    poa = np.clip(clear * cloud + 0.01 * rng.standard_normal(n) * (clear > 0), 0, None)
    ghi = poa * 0.92
    temp = 14 + 9 * np.sin(2 * np.pi * (sec - 9 * 3600) / 86400) + 1.5 * smooth_noise(3600, n)
    ws = np.clip(3.5 + 2.2 * smooth_noise(1800, n) + 0.8 * smooth_noise(15, n), 0, None)
    wd = np.mod(180 + 60 * smooth_noise(1200, n) + 10 * smooth_noise(10, n), 360)
    hum = np.clip(0.65 - 0.2 * np.sin(2 * np.pi * (sec - 9 * 3600) / 86400) + 0.05 * smooth_noise(1800, n), 0.1, 1.0)
    pres = 1010 + 4 * smooth_noise(6 * 3600, n)

    PV, _ = import_PV_WT_data()
    pac, _, _, _ = pv_model_arrays(poa, temp, ws, PV)
    meas = pac * np.clip(0.92 + 0.04 * smooth_noise(60, n), 0.6, 1.05)
    curtail = smooth_noise(2400, n) > 1.1                                           # curtailment-like blocks
    meas = np.where(curtail, meas * 0.3, meas)
    meas = np.where(poa <= 0.0005, np.abs(0.002 * rng.standard_normal(n)) * (rng.random(n) < 0.3), meas)
    gaia = np.clip(2.0 * np.clip(ws - 3.5, 0, None) ** 1.5 * (rng.random(n) < 0.02), 0, 11)

    df = pd.DataFrame({
        'TEMPERATURE[degC]': temp, 'HUMIDITY[%]': hum, 'WIND_SPEED[m1s]': ws, 'WIND_DIR[deg]': wd,
        'GHI[kW1m2]': ghi, 'POA Irr[kW1m2]': poa, 'P_Gaia[kW]': gaia, 'P_Solar[kW]': meas,
        'Pressure[mbar]': pres}, index=idx1)
    if with_az_el:
        df['Azimuth[deg]'] = np.mod(sec / 86400 * 360 - 180, 360) - 180
        df['Elevation[deg]'] = 60 * np.sin(np.pi * (sec - 4.5 * 3600) / (15 * 3600))
    if step > 1:   # bucket means, labelled by interval start (the v4 convention)
        df = df.resample(f'{step}s', label='left', closed='left').mean()
    return df
