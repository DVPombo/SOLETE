# -*- coding: utf-8 -*-
"""
Part of the solete_pipeline package -- split out of the original
solete/ (Phase 7, Session 8) for independent testability.
See solete/ (kept as a re-export shim) and CONTRIBUTING.md for
why this split happened and how the modules relate to each other.

Original author: Daniel Vázquez Pombo
email: daniel.vazquez.pombo@gmail.com

Licensed under the MIT License -- see LICENSE at the repo root. If you use this work, please give credit (see CITATION.cff).
"""

import pandas as pd
import numpy as np
import sys

# CoolProp is imported lazily inside Rincon_Pombo_ThermodynamicModel (the only user), so that the PV model,
# solete.expansion and the dataset build do not need it installed.


def PV_Performance_Model(data, PVinfo, colirra='POA Irr[kW1m2]', coltemp='TEMPERATURE[degC]',colwindspeed='WIND_SPEED[m1s]'):
    """
    This function implements King's PV performance model. More info in [2].
    
    Parameters
    ----------
    data : DataFrame
        Variable including all the data from the Solete dataset
    PVinfo : dict
        A bunch of parameters extracted from the datasheet and other supporting documents
        Check function: import_PV_WT_data for further details
    colirra : string, optional
        holds Epoa, that is the irradiance in the plane of the array in kW/m2
        If you reuse this code, make sure you are feeding Epoa and not GHI
        The default is 'POA Irr[kW1m2]'.
    coltemp : string, optional
        holds the ambient temperature in Celsius.
        The default is 'TEMPERATURE[degC]'.
    colwindspeed : string, optional 
        holds the wind speed in m/sec
        The default is'WIND_SPEED[m1s]'

    Returns
    -------
    DataFrames
        Pac, Pdc, Tm, and Tc. 

    """
    
    
    Pac, Pdc, Tm, Tc = pv_model_arrays(data[colirra].to_numpy(), data[coltemp].to_numpy(),
                                       data[colwindspeed].to_numpy(), PVinfo)
    return (pd.Series(Pac, index=data.index), pd.Series(Pdc, index=data.index),
            pd.Series(Tm, index=data.index), pd.Series(Tc, index=data.index))


def _nansum2(a, b):
    """Element-wise sum of two arrays skipping NaN, all-NaN -> 0 (pandas `DataFrame.sum(axis=1)`)."""
    return np.where(np.isnan(a), 0.0, a) + np.where(np.isnan(b), 0.0, b)


def _nanmean2(a, b):
    """Element-wise mean of two arrays skipping NaN, all-NaN -> NaN (pandas `DataFrame.mean(axis=1)`)."""
    na, nb = np.isnan(a), np.isnan(b)
    total = np.where(na, 0.0, a) + np.where(nb, 0.0, b)
    count = (~na).astype(np.float64) + (~nb).astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(count > 0, total / np.where(count > 0, count, 1.0), np.nan)


def pv_model_arrays(irradiance, temperature, wind_speed, PVinfo):
    """
    King's PV performance model on plain NumPy arrays: strictly row-wise, no dependence on
    neighbouring rows, no assumption about the time step, so it can be applied to any slice of
    any resolution and gives the same numbers for a row whatever slice it is in.

    This is the arithmetic of the original `PV_Performance_Model` (same operations in the same
    order; verified bit-for-bit against a frozen copy of the old implementation in
    tests/test_expansion.py), without the per-call DataFrame of ~20 intermediate columns.
    Behaviour kept on purpose: a row with a NaN input gives Pac = Pdc = 0 (the old
    `DataFrame.sum(axis=1)` skipped NaN) and NaN temperatures.

    Parameters: irradiance in kW/m2 (plane of array), temperature in degC, wind speed in m/s,
    all 1-D arrays of the same length. Returns (Pac_kW, Pdc_kW, Tm_degC, Tc_degC).
    """
    irr = np.asarray(irradiance, dtype=np.float64)
    temp = np.asarray(temperature, dtype=np.float64)
    ws = np.asarray(wind_speed, dtype=np.float64)

    Pac_ch, Pmp_ch, Tm_ch, Tc_ch = [], [], [], []
    for i in range(len(PVinfo['index'])):
        a, b, d_t = PVinfo['a'][i], PVinfo['b'][i], PVinfo['D_T'][i]
        eff_max = max(PVinfo['eff_%'][i])      # maximum inverter efficiency in %
        p_inv_max = max(PVinfo['eff_P'][i])    # W, maximum power output of the inverter
        n_panels = PVinfo['Ns'][i] * PVinfo['Np'][i]

        irr_w = irr * 1000
        Tm = temp + irr_w * np.exp(a + b * ws)                                  # module temperature
        Tc = Tm + irr_w / PVinfo["Estc"] * d_t                                  # cell temperature
        Pmp_panel = irr_w / PVinfo["Estc"] * PVinfo['Pmp_stc'][i] * (
            1 + PVinfo['ganma_mp'][i] * (Tc - PVinfo["Tstc"]))                  # one panel
        Pmp_array = n_panels * Pmp_panel                                        # whole array
        Pac = eff_max / 100 * Pmp_array
        # clamp to the inverter capacity, then to >= 0 (NaN stays NaN in both comparisons)
        Pac = np.where(Pac > p_inv_max, float(p_inv_max), Pac)
        Pac = np.where(Pac < 0, 0.0, Pac)
        Pac_ch.append(Pac); Pmp_ch.append(Pmp_array); Tm_ch.append(Tm); Tc_ch.append(Tc)

    if len(Pac_ch) != 2:
        raise ValueError("pv_model_arrays expects the two inverter channels 'A' and 'B' of PVinfo")
    return (_nansum2(*Pac_ch) / 1000, _nansum2(*Pmp_ch) / 1000,
            _nanmean2(*Tm_ch), _nanmean2(*Tc_ch))

def Rincon_Pombo_ThermodynamicModel(data, pv, verbose=0):
    """
    This function implements section 4.4 from [4]. That is, an advanced thermodynamic 
    performance model for photovoltaic pannels. All credit for the coding goes to my
    good friend Mario Javier Rincón Pérez (mjrp@mpe.au.dk). I simply adapted it to fit
    in the SOLETE platform. If you are into fluid and thermodynamics reach out to him.

    Phase 7 Session 2 performance note: the physical model and its exact per-row
    behavior (including the gradient-limiter quirk described below) are unchanged
    from the original port -- this is a speed-only refactor, validated to produce
    bit-for-bit identical output against the original row-by-row implementation on
    both real SOLETE files. See CHANGELOG.md for before/after timing and what was
    changed. Summary of what moved: everything that depends only on a row's own
    (T, p, humidity, wind_speed) -- and nothing recursive -- is computed once for
    all rows up front. That includes the three `CoolProp.HAPropsSI` calls (mu, cp,
    k), which `HAPropsSI` accepts as array arguments directly in the pinned
    CoolProp 8.0.0 (replacing 3*N scalar Python-level calls with 3 calls total),
    and the elementwise arithmetic (density, Reynolds/Prandtl numbers, the flat-
    plate Rex grid). The one piece deliberately left as a loop even though it
    doesn't depend on T_PV is the power-law/branch part of the 100-point
    discretization (see the comment at that loop for why: numpy's vectorized `**`
    is not bit-identical to Python's scalar `**` here) -- it's cheap compared to
    the CoolProp calls, so looping it costs little and keeps the output exact.
    What genuinely can't be hoisted out stays in the loop below exactly as
    before: the radiative term and the natural-convection override both depend on
    the PV temperature carried over from the previous row (`T_PV`), and the
    gradient limiter depends on the model's own temperature history.

    Parameters
    ----------
    data : DataFrame
            Variable including all the data from the Solete dataset
    verbose : int, optional
        The default is 0.
        If a 1 is fed, the iterations are shown. 

    Returns
    -------
    df : DataFrame
        Temperature of the modules according to the Rincón-Pombo method [4].

    """

    print("")    
    print("Expanding SOLETE with the Rincon-Pombo thermodynamic model")    
    print("    This is going to take a while be patient.")    
    # Hardcoded INPUTS
    g = 9.81  # gravity m/s^2
    psi = 0.1  # Gradient limiter factor
    gradLimiter = 5  # max gradT allowed without limiter applied
    # PV cells
    # E_STC = 1000  # Solar irradiance W/m^2
    E_POA = data['POA Irr[kW1m2]'].to_numpy() * 1000  # Solar irradiance W/m^2
    # E_POA = data['GHI[kW/m2]'] * E_STC
    epsilon = 0.3  # radiative emissivity (glass)
    SB = 5.670374419e-8  # stefan boltzmann constant
    reflectitivy = 0.6  # light that is reflected by the module to the atmoshpere \reflactance
    transmittance = 0.1  # these three sum  \tau trasmittance
    absorption = 1 - reflectitivy - transmittance  # \alpha absorptance
    IRratio = 0.53  # Infra Red light factor = contributes to heating
        
    # Air
    # (the original ideal-gas constant `R = 8.31432e3` declared here was never
    # actually read before being overwritten by the per-row thermal-resistance
    # `R` further down -- dropped as genuinely dead code, not a behavior change.)
    R_a = 285.9  # dry air gas constant
    R_w = 461.5  # water vapour constant
    T = (273.15 + data['TEMPERATURE[degC]']).to_numpy()  # dry bulb temperature K
    p = data['Pressure[mbar]'].to_numpy() * 100  # pressure Pa
    #plain numpy arrays for everything indexed by [i] below: these Series carry the
    #dataset's DatetimeIndex, and pandas no longer allows integer keys like [i] to fall back to
    #positional access on a non-integer index -- it now always treats them as (nonexistent) labels
    #and raises KeyError. Working on arrays makes the intended positional access unambiguous.
    humidity = data['HUMIDITY[%]'].to_numpy().copy()
    wind_speed = data['WIND_SPEED[m1s]'].to_numpy()
    N = len(data)
    A = pv["L"] * pv["W"]  # PV area

    # ------------------------------------------------------------------
    # Vectorized precompute: every quantity below depends only on this row's
    # own (T, p, humidity, wind_speed) -- never on T_PV -- so it is computed
    # once for all N rows up front instead of once per iteration.
    # ------------------------------------------------------------------
    # Same clip as the original (`if humidity[i] > 1: humidity[i] = 1.0`),
    # applied to the whole array before it's used anywhere below -- the
    # original clip happens before humidity[i]'s first use each iteration too,
    # so this is equivalent, not just similar.
    humidity = np.where(humidity > 1, 1.0, humidity)

    rho_a = p / (R_a * T)  # density of dry air
    rho = rho_a * (1 + humidity) / (1 + R_w / R_a * humidity)  # density of mixture

    # CoolProp 8.0.0's HAPropsSI accepts array arguments directly (verified
    # against a scalar-loop call on the same inputs -- bit-for-bit identical,
    # not merely close), so this is 3 calls total instead of 3*N.
    from CoolProp.HumidAirProp import HAPropsSI   # lazy: see top of file
    mu = HAPropsSI('mu', 'P', p, 'T', T, 'R', humidity)  # dynamic viscosity
    cp = HAPropsSI('cp_ha', 'P', p, 'T', T, 'R', humidity)  # specific heat per unit of humid air
    k = HAPropsSI('k', 'P', p, 'T', T, 'R', humidity)  # thermal conductivity
    nu = mu / rho
    beta = 1 / T  # thermal expansion coefficient for ideal gases

    Re = rho * np.abs(wind_speed) * pv["L"] / mu  # Reynolds number (per row)
    Pr = cp * mu / k  # Prandtl number (per row)

    # Vectorized flat-plate discretization (was a 100-point Python loop per
    # row). `x_grid` is the exact same grid the original built with
    # `np.linspace(0, pv["L"], num=100)`; Rex is now an (N, 100) array instead
    # of one scalar recomputed 100 times per row -- elementwise multiply/
    # divide is IEEE-754 exact regardless of whether it's done as a scalar or
    # as a batched numpy operation, so this part is safe to vectorize fully.
    x_grid = np.linspace(0, pv["L"], num=100)
    Rex = rho[:, None] * np.abs(wind_speed)[:, None] * x_grid[None, :] / mu[:, None]

    # Nux itself is NOT vectorized the same way, on purpose: numpy's ufunc for
    # `**` on a sizable array uses a SIMD-approximated power that can differ
    # from Python/libm's scalar `pow()` by ~1 ULP (confirmed directly: e.g.
    # `Pr**(1/3)` computed as a 5000-element array differs from the same
    # values computed one at a time in ~6-7% of elements, by up to 1 ULP;
    # indexing back down to individual scalars, as the loop below does, uses
    # the same precise scalar path the original per-row loop did). That 1-ULP
    # noise is small in itself, but the loop below feeds back into `T_PV`
    # every iteration, and re-running this same check with the fully
    # vectorized `**` showed it compounds to a ~5.7e-14 absolute (~1.9e-16
    # relative) difference by the end of a ~11,000-row file -- close to
    # machine epsilon, but not the bit-for-bit match this refactor is
    # supposed to preserve. So only the cheap part (Rex; ordinary +-*/) is
    # vectorized, and the power-law/branch part -- 100 scalar-typed
    # iterations per row, not a CoolProp call -- stays a loop, which keeps
    # bit-for-bit equivalence with the original at a small, measured cost
    # (still far cheaper than the per-row CoolProp calls this refactor
    # actually targets).
    hx = np.empty((data.shape[0], 100))
    for i in range(data.shape[0]):
        Pr_i = Pr[i]
        k_i = k[i]
        row = Rex[i]
        for j in range(100):
            Rex_ij = row[j]
            if Rex_ij <= 1e5:  # laminar, similarity solutions
                if Pr_i < 0.6:  # never observed on real SOLETE data; kept for parity
                    print('Correlation does not satisfy')
                Nux = 0.453 * Rex_ij ** (1 / 2) * Pr_i ** (1 / 3)
            else:  # turbulent, empirical correlations
                Nux = 0.0308 * Rex_ij ** (4 / 5) * Pr_i ** (1 / 3)
            hx[i, j] = 0.0 if j == 0 else Nux * k_i / x_grid[j]

    h_forced = np.mean(hx, axis=1)  # mean convective heat transfer coefficient (W/m^2/K), per row

    d_over_2kA = pv["d"] / (pv["k_r"] * A * 2)  # constant term reused every iteration below

    # ------------------------------------------------------------------
    # Sequential loop: only what genuinely depends on the running PV
    # temperature (T_PV) and the model's own temperature history stays here.
    # No CoolProp calls and no per-row discretization loop remain in this
    # part -- everything above already covers what those needed.
    # ------------------------------------------------------------------
    T_plot = np.empty(N)
    gradT = np.empty(N)
    T_PV = 273.15 + data['TempModule'].iloc[0]  # initialise temperature of PV cell

    for i in range(N):
        # RADIATION
        if T_PV >= T[i]:
            q_epsilon = -epsilon * SB * A * (T_PV ** 4 - T[i] ** 4)
        else:
            q_epsilon = 0

        q_absorbed = E_POA[i] * A * IRratio * absorption

        # CONVECTION -- forced-convection h precomputed above; only the
        # natural-convection override (Gr/Ra depend on T_PV) is computed here.
        Gr = g * beta[i] * abs((T_PV - T[i])) * (A / (2 * pv["W"] + 2 * pv["L"])) ** 3 / nu[i] ** 2  # Grashof number
        Ra = Gr * Pr[i]  # Rayleigh number

        h = h_forced[i]

        if 1e4 < Ra < 1e7 and Re[i] < 1e3:  # Natural convection, empirical correlations
            Nu = 0.54 * Ra ** (1 / 4)
            h = Nu * k[i] / pv["L"]  # mean convective heat transfer coefficient from correlations (W/m^2/K)

        elif 1e7 < Ra < 1e11 and Re[i] < 1e3:  # Natural convection, empirical correlations
            Nu = 0.15 * Ra ** (1 / 3)
            h = Nu * k[i] / pv["L"]  # mean convective heat transfer coefficient from correlations (W/m^2/K)

        if h == 0:  # numerical solution for problems in convection or inputs
            h = 1e-16
            R = d_over_2kA  # Thermal resistance of the system
        else:
            R = 1 / (h * A) + d_over_2kA  # Thermal resistance of the system

        q_convection = h * A * (T[i] - T_PV)

        # Heat balance
        q = (q_absorbed + q_convection + q_epsilon)
        T_PV = T_PV + q * R

        if i == 0:
            gradT[i] = 0
        else:
            # Preserved exactly, including its original quirk: when the
            # limiter trips, it overwrites the *previous* row's stored
            # temperature (`T_plot[i - 1]`) with the current, now-limited
            # T_PV, which makes the just-computed gradient collapse back to
            # 0 (since it becomes `T_PV - T_PV`). That is how the original
            # `np.append`-based code actually behaved (`T_plot[-1]` referred
            # to the previous row until this row's value was appended a few
            # lines later) -- this is a speed refactor, not a correctness
            # one, so that behavior is kept rather than "fixed" here.
            gradT[i] = T_PV - T_plot[i - 1]
            if abs(gradT[i]) >= gradLimiter:  # Gradient limiter function
                T_PV = T_plot[i - 1] + psi * gradT[i]
                T_plot[i - 1] = T_PV
                gradT[i] = T_PV - T_plot[i - 1]

        T_plot[i] = T_PV

        if i % 500 == 0: #output progress every 500 samples
            msg='    Progress: ' + str(round(i/N*100)) + ' %'
            sys.stdout.write('\r'+msg)

    df = T_plot - 273.15
    
    msg='    Progress: ' + str(100) + ' %'
    sys.stdout.write('\r'+msg)    
    print("")
    
    
    return df

